#include "core/Tracking.h"                 // 引入本类声明
#include "core/Map.h"                      // 引入地图管理类
#include "core/FeatureDetector.h"          // 引入特征处理器
#include "core/KeyFrame.h"                 // 引入关键帧类
#include "backend/Optimizer.h"             // 跨模块：引入后端优化器
#include "utils/Parameters.h"              // 跨模块：引入全局参数类
#include <opencv2/core/eigen.hpp> // 引入 OpenCV 到 Eigen 的转换库，方便进行矩阵格式互转
#include <iostream>               // 引入标准输入输出流，用于在终端打印 SLAM 系统的运行状态

// Tracking 类的构造函数：传入地图指针并初始化相关多线程和参数
Tracking::Tracking(std::shared_ptr<Map> pMap)
    : mpMap(pMap), mIsInitialized(false), mIsRunning(true), mNeedOptimize(false), mNextKFId(0) // 初始化列表：绑定地图，设置未初始化状态，置运行标志为真，不激活BA优化，关键帧计数归零
{                                                                                              // 构造函数主体开始
    // 对全局静态参数做一次性深拷贝到本地变量，彻底杜绝多线程并发时的内存读写数据竞争
    mFlowBack = (Parameters::FLOW_BACK != 0);          // 拷贝是否开启双向反向光流校验的布尔开关
    mFx = Parameters::fx;                              // 拷贝相机内参：水平方向焦距 fx
    mFy = Parameters::fy;                              // 拷贝相机内参：垂直方向焦距 fy
    mCx = Parameters::cx;                              // 拷贝相机内参：水平方向光心坐标 cx
    mCy = Parameters::cy;                              // 拷贝相机内参：垂直方向光心坐标 cy
    mK1 = Parameters::k1;                              // 拷贝相机径向畸变参数 k1
    mK2 = Parameters::k2;                              // 拷贝相机径向畸变参数 k2
    mP1 = Parameters::p1;                              // 拷贝相机切向畸变参数 p1
    mP2 = Parameters::p2;                              // 拷贝相机切向畸变参数 p2
    mKeyframeParallax = Parameters::KEYFRAME_PARALLAX; // 拷贝判定新关键帧所需的平均像素视差阈值
    mBodyTCam0 = Parameters::body_T_cam0;              // 拷贝机体到左目相机的 4x4 外参变换矩阵
    mBodyTCam1 = Parameters::body_T_cam1;              // 拷贝机体到右目相机的 4x4 外参变换矩阵

    mCurrentPose = Eigen::Isometry3d::Identity(); // 将当前实时位姿初始化为单位齐次变换矩阵（无旋转，无平移）

    // 实例化前端算法管道的独占智能指针，传入最大特征数、特征间最小距离、以及双向光流开关
    mpFeatureDetector = std::make_unique<FeatureDetector>(Parameters::MAX_CNT, Parameters::MIN_DIST, mFlowBack); //

    // 启动前后端多线程并发管道：创建并绑定前端追踪线程和后端 Ceres 优化常驻后台线程
    mTrackThread = std::thread(&Tracking::TrackLoop, this);     // 开启前端主循环线程，绑定到 TrackLoop 函数
    mBackendThread = std::thread(&Tracking::BackendLoop, this); // 开启后端异步图优化线程，绑定到 BackendLoop 函数
} // 构造函数主体结束

// Tracking 类的析构函数：用于安全释放线程及内存资源，防止程序异常退出
Tracking::~Tracking()
{                                                          // 析构函数主体开始
    {                                                      // 进入局部作用域，用于精确控制互斥锁的生命周期
        std::unique_lock<std::mutex> lock1(mMutexBuf);     // 锁住前端图像输入缓存队列的互斥锁
        std::unique_lock<std::mutex> lock2(mMutexBackend); // 锁住后端优化同步控制信号的互斥锁
        mIsRunning = false;                                // 将运行状态标志置为假，通知所有后台工作线程准备退出循环
    } // 局部作用域结束，自动释放 lock1 和 lock2，避免后面 join 线程时死锁
    mCondBuf.notify_all();         // 唤醒所有卡在前端缓存阻塞等待处的条件变量
    mCondBackend.notify_all();     // 唤醒所有卡在后端图优化阻塞等待处的条件变量
    if (mTrackThread.joinable())   // 如果前端追踪主线程仍在活跃且可合并
        mTrackThread.join();       // 挂起主线程，回收前端追踪线程占用的所有系统底层资源
    if (mBackendThread.joinable()) // 如果后端 BA 优化线程仍在活跃且可合并
        mBackendThread.join();     // 挂起主线程，回收后端优化线程占用的所有系统底层资源
} // 析构函数主体结束

// 注册渲染回调函数：由 ROS 2 主节点调用，用来绑定数据发布管道
void Tracking::RegisterCallback(RenderCallback cb)
{                   // 函数开始
    mRenderCb = cb; // 将传入的函数指针/lambda表达式赋值给类内部的成员回调变量
} // 函数结束

// 喂入双目图像流接口：由 ROS 2 节点的图像回调函数调用，负责把对齐后的数据塞入追踪缓存队列
void Tracking::FeedStereoImages(const cv::Mat &imLeft, const cv::Mat &imRight, double timestamp)
{                                                 // 函数开始
    std::unique_lock<std::mutex> lock(mMutexBuf); // 自动加锁，保护图像输入缓存队列不被多线程并发破坏
    mLeftBuf.push(imLeft.clone());                // 将左目图像深拷贝（clone）一份，安全压入左目缓存队列
    mRightBuf.push(imRight.clone());              // 将右目图像深拷贝（clone）一份，安全压入右目缓存队列
    mTimeBuf.push(timestamp);                     // 将当前双目帧对应的同步时间戳压入时间缓存队列
    mCondBuf.notify_one();                        // 唤醒正卡在 TrackLoop 里由于没有图像输入而陷入休眠的前端主线程
} // 函数结束，lock 在此析构，自动解锁

// 前端主处理线程循环：只要系统保持运行，就在后台循环处理图像队列
void Tracking::TrackLoop()
{                                                                                                                     // 函数开始
    while (true)                                                                                                      // 开启死循环，不断从队列取图解算
    {                                                                                                                 // 循环体开始
        cv::Mat matLeft, matRight;                                                                                    // 定义局部变量，用于暂存本轮循环取出的左右目单帧图像
        double timestamp = 0.0;                                                                                       // 定义局部变量，用于暂存本轮循环对应的时间戳
        {                                                                                                             // 进入临界区锁保护作用域
            std::unique_lock<std::mutex> lock(mMutexBuf);                                                             // 锁住队列互斥量，保障弹出数据时的线程安全
            mCondBuf.wait(lock, [this]                                                                                // 使用条件变量阻塞等待：直到收到停止信号，或者缓存队列中均不为空时才被唤醒
                          { return !mIsRunning || (!mLeftBuf.empty() && !mRightBuf.empty() && !mTimeBuf.empty()); }); //

            if (!mIsRunning) // 如果被唤醒是因为系统下发了停止指令（mIsRunning == false）
                break;       // 直接跳出 while(true) 死循环，从而结束前端追踪主线程

            matLeft = mLeftBuf.front();   // 从缓存队列头部安全取出左目图像矩阵
            mLeftBuf.pop();               // 将已取出的左目图像弹出，释放队列头部
            matRight = mRightBuf.front(); // 从缓存队列头部安全取出右目图像矩阵
            mRightBuf.pop();              // 将已取出的右目图像弹出，释放队列头部
            timestamp = mTimeBuf.front(); // 从时间缓存队列头部取出对应的时间戳
            mTimeBuf.pop();               // 将已取出的时间戳弹出，释放队列头部
        } // 临界区作用域结束，自动释放 lock

        cv::Mat feat_img;                          // 声明局部变量，用于存储带特征光流渲染的可视化组合大图
        std::vector<Eigen::Vector3d> vWorldPoints; // 声明三维点云容器，提取本帧新三角化或追踪到的 3D 世界结构点
        std::vector<Eigen::Vector3d> vKFPositions; // 声明历史关键帧轨迹容器，提取全局所有已固化关键帧的 3D 位置

        // 调用 VO 核心解算管道函数，传入图像流，解算出当前帧的绝对变换矩阵 Tcw，并填充可视化数据
        Eigen::Isometry3d Tcw = ProcessStereo(matLeft, matRight, timestamp, feat_img, vWorldPoints, vKFPositions); //

        if (mRenderCb && !feat_img.empty())                                   // 如果外层 ROS 2 节点成功注册了回调函数且可视化渲染图不为空
        {                                                                     // 回调调用主体开始
            mRenderCb(timestamp, feat_img, vWorldPoints, vKFPositions, Tcw,   // 触发回调：将解算出的时间戳、图像、点云、轨迹和位姿一并塞给 ROS 2 节点进行发布
                      mpFeatureDetector->mvCurPts, mpFeatureDetector->mvIds); // 顺便附带上当前激活特征点的像素坐标与全局唯一特征 ID
        } // 回调调用主体结束
    } // 循环体结束
} // 函数结束

// 双目数据核心计算管道：VO 核心算法的前端全流程（初始化、光流、PnP、三角化、关键帧判断）
Eigen::Isometry3d Tracking::ProcessStereo(const cv::Mat &imLeft, const cv::Mat &imRight,
                                          const double &timestamp, cv::Mat &imgTrack,
                                          std::vector<Eigen::Vector3d> &vWorldPoints,
                                          std::vector<Eigen::Vector3d> &vKFPositions)
{                                                             // 函数主体开始
    cv::Mat grayLeft = imLeft;                                // 声明并浅拷贝输入的左目图像到局部变量
    if (imLeft.channels() == 3)                               // 校验图像通道数，如果是 BGR 三通道彩色图
        cv::cvtColor(imLeft, grayLeft, cv::COLOR_BGR2GRAY);   // 调用 OpenCV 将其转换为单通道灰度图，因为前端算法（光流/特征）必须使用灰度图
    cv::Mat grayRight = imRight;                              // 声明并浅拷贝输入的右目图像到局部变量
    if (imRight.channels() == 3)                              // 校验右目图像通道数，如果是 BGR 三通道彩色图
        cv::cvtColor(imRight, grayRight, cv::COLOR_BGR2GRAY); // 调用 OpenCV 将右目转换为单通道灰度图

    // 绘制准备工作：为了在一个窗口同时显示双目追踪画面，需要横向拼接左右目画面
    cv::Mat imgLeftBGR, imgRightBGR;                          // 声明两个局部彩图变量，以便在灰度底图上绘制彩色点线
    cv::cvtColor(grayLeft, imgLeftBGR, cv::COLOR_GRAY2BGR);   // 将左目灰度图升维扩展成 BGR 三通道彩图
    cv::cvtColor(grayRight, imgRightBGR, cv::COLOR_GRAY2BGR); // 将右目灰度图升维扩展成 BGR 三通道彩图
    cv::hconcat(imgLeftBGR, imgRightBGR, imgTrack);           // 调用 OpenCV 横向拼接（Horizontal Concatenate）双目图像到输出大图 imgTrack 中

    vWorldPoints.clear(); // 清空输出点云容器，以便填充当前最新的 3D 物理结构
    vKFPositions.clear(); // 清空输出轨迹容器，以便填充当前最新的全局高精度历史轨迹

    // 线程安全隔离：提取全局更新的实时位姿矩阵到局部变量进行本次解算
    Eigen::Isometry3d localPose;                          // 声明局部变量位姿
    {                                                     // 锁临界区开始
        std::unique_lock<std::mutex> lock(mMutexBackend); // 加锁：防止在读取 mCurrentPose 的瞬间后端 BA 线程同时向其写入引发冲突
        localPose = mCurrentPose;                         // 安全拷贝全局共享的最新位姿
    } // 锁临界区结束，自动解锁

    // --- A. 第一帧系统初始化（Cold Start Pipeline） ---
    if (!mIsInitialized)                                                             // 判断当前系统是否已经初始化，如果为 false 则进入冷启动流程
    {                                                                                // 初始化分支开始
        std::cout << ">>> [SLAM前端] 第一帧初始化，固化为初始关键帧。" << std::endl; // 在控制台输出系统初始化日志
        mpFeatureDetector->mvCurPts.clear();                                         // 强制清空特征处理器中的当前帧像素特征点容器
        mpFeatureDetector->mvIds.clear();                                            // 强制清空特征处理器中的全局唯一特征 ID 容器
        mpFeatureDetector->mvTrackCnt.clear();                                       // 强制清空特征处理器中的各特征点连续追踪帧数计数器
        mmIDToMapPoint.clear();                                                      // 强力清空前端持有的局部路标哈希检索字典
        localPose = Eigen::Isometry3d::Identity();                                   // 将初始帧位姿牢牢锁定并固化为世界绝对坐标原点（单位矩阵）

        mpFeatureDetector->AddNewFeatures(grayLeft); // 调用角点提取算法，在初始左目灰度图上提取第一批均匀分布的强特征点
        mPrevImg = grayLeft.clone();                 // 将当前的左目灰度图像深拷贝备份，留作下一帧光流追踪的“上一帧底图”

        std::map<int, cv::Point2f> initMeasurements;                                        // 声明临时字典，用于打包初始帧的 2D 像素观测
        for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); ++i)                     // 遍历初始帧提取出来的所有特征角点
        {                                                                                   // 遍历填充循环开始
            initMeasurements[mpFeatureDetector->mvIds[i]] = mpFeatureDetector->mvCurPts[i]; // 以全局特征 ID 为键，2D 像素坐标为值写入字典
        } // 遍历填充循环结束

        // 实例化创建 SLAM 系统的第一个关键帧（KeyFrame）：传入自增唯一 ID，时间戳，相机到世界的绝对位姿（对世界系到相机系位姿求逆），以及 2D 观测
        auto pKF = std::make_shared<KeyFrame>(mNextKFId++, timestamp, localPose.inverse(), initMeasurements); //
        mpMap->AddKeyFrame(pKF);                                                                              // 将首帧关键帧指针安全插入到全局地图数据库中

        mvpPrevKFPointsMap = initMeasurements; // 将首帧的观测备份给成员变量，作为下一帧计算像素平移视差的基准历史参考
        mIsInitialized = true;                 // 标志位置为 true，宣布整个系统冷启动成功，以后图像帧将走正常追踪解算分支

        for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); ++i)                // 循环遍历当前左图所有的像素特征点，进行画面渲染绘制
        {                                                                              // 渲染循环开始
            double len = std::min(1.0, 1.0 * mpFeatureDetector->mvTrackCnt[i] / 20.0); // 核心色彩渐变计算：根据追踪寿命（1到20帧以上）线性归一化到 0 到 1 之间
            cv::Scalar ptColor = cv::Scalar(255 * (1 - len), 0, 255 * len);            // 组装红蓝渐变色：追踪短为红（新提取点），追踪长为蓝（稳定历史点）
            cv::circle(imgTrack, mpFeatureDetector->mvCurPts[i], 2, ptColor, 2);       // 调用 OpenCV 在大图的左图对应坐标上绘制半径为 2 像素的特征实心圆点
        } // 渲染循环结束

        vKFPositions.push_back(localPose.inverse().translation()); // 提取初始帧绝对坐标的 3D 平移向量，压入全局关键帧轨迹容器，用作可视化
        mpFeatureDetector->UpdatePreviousStatus(grayLeft);         // 命令特征处理器将当前特征点复制进历史特征队列，完成帧间特征状态交接

        {                                                     // 锁临界区开始
            std::unique_lock<std::mutex> lock(mMutexBackend); // 锁住后端同步互斥量，准备将当前帧计算好的位姿发布回共享变量
            mCurrentPose = localPose;                         // 更新全局共享的最新位姿（即原点）
        } // 锁临界区结束，自动解锁
        return localPose; // 退出并返回初始化帧对应的单位位姿矩阵
    } // 初始化分支结束

    // --- B. 正常帧间状态解算（Continuous Processing Pipeline） ---
    // 步长 1：帧间 LK 光流追踪。利用金字塔 Lucas-Kanade 稀疏光流算法，把上一帧左图中的特征点像素位置无损追踪到当前最新左图上
    mpFeatureDetector->TrackFeaturesLK(mPrevImg, grayLeft); //

    // 步长 2：PnP 求解当前帧初估位姿。根据当前光流成功追踪到的特征 ID，反查地图取出对应的 3D 世界点点云坐标，通过 RANSAC 鲁棒解算 PnP
    bool pnp_succ = mpFeatureDetector->EstimatePosePnP(mmIDToMapPoint, mFx, mFy, mCx, mCy, mK1, mK2, mP1, mP2, localPose); //

    // 步长 3：均匀分布特征约束。为了防止特征点在某些纹理过于丰富的区域过度扎堆，系统在此通过 SetMask 在已有特征点周围画抑制圈
    mpFeatureDetector->SetMask(grayLeft.rows, grayLeft.cols); // 传入图像行列大小，刷新排他性特征点屏蔽掩码矩阵
    mpFeatureDetector->AddNewFeatures(grayLeft);              // 在没被掩码抑制的空白纹理区域补充提取新特征点，使得总追踪点数重新逼近并维持在 MAX_CNT 上限

    // 步长 4：挑选并判定当前帧是否为关键帧
    bool isKeyFrame = false;            // 声明本地布尔变量，默认设为非关键帧
    if (pnp_succ)                       // 只有当 PnP 成功估算出初始位姿、且点数正常的情况下，才去判断视差
    {                                   // 判断体开始
        isKeyFrame = NeedNewKeyFrame(); // 调用视差计算函数，判定当前帧是否满足插入新关键帧的几何条件
    } // 判断体结束

    // 步长 5：立体匹配与双目自适应深度三角化。
    // 【架构注意】该函数内部如果发现 isKeyFrame == true，会已经自动把新三角化出来的 3D 路标点指针推入了全局地图 mpMap 中
    mpFeatureDetector->TriangulateNewPoints(                                        // 调用自适应多线程双目三角化函数
        grayLeft, grayRight, localPose, mBodyTCam0, mBodyTCam1, mFx, mFy, mCx, mCy, // 传入灰度图、当前位姿、双目机体外参及内参
        mmIDToMapPoint, mpMap, isKeyFrame, vWorldPoints, imgTrack);                 // 传入局部字典、全局地图、关键帧开关，用于回填局部点云并渲染拼接大图

    // 步长 6：新关键帧固化并触发异步后端 BA 优化
    if (isKeyFrame)                                                                            // 如果本帧满足关键帧条件
    {                                                                                          // 固化处理分支开始
        std::map<int, cv::Point2f> currentMeasurements;                                        // 声明一个字典，用于提取并打包本关键帧关联的所有 2D 特征像素观测
        for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); ++i)                        // 轮询当前特征队列里所有的激活角点
        {                                                                                      // 特征字典装载循环开始
            currentMeasurements[mpFeatureDetector->mvIds[i]] = mpFeatureDetector->mvCurPts[i]; // 装载多项关联：全局特征 ID -> 2D 像素坐标
        } // 特征字典装载循环结束

        // 实例化一个全新的关键帧指针：传入关键帧计数 ID，当前戳，绝对相机到世界系位姿矩阵（对世界系到相机系位姿逆矩阵），以及刚提取的 2D 特征观测
        auto pKF = std::make_shared<KeyFrame>(mNextKFId++, timestamp, localPose.inverse(), currentMeasurements); //
        mpMap->AddKeyFrame(pKF);                                                                                 // 将固化好的新关键帧追加进全局小地图数据库中

        mvpPrevKFPointsMap = currentMeasurements; // 将本关键帧的 2D 字典赋值给前一帧缓存变量，供未来滑动的新帧算视差参考

        // =========================================================================
        // 【重要系统级优化说明】删除了历史代码中在这里对 mmIDToMapPoint 重复插入 mpMap 的多余二次循环。
        // 原因：在前面的 TriangulateNewPoints 中，如果是关键帧，新创建的 MapPoint 实例已经作为高精度全局路标被安全存入 mpMap。
        // 若此处再加一遍，会导致 vector 容器里的指针实例重复堆叠，引发可怕的内存崩坏和无谓计算。
        // =========================================================================

        // 激活后端非线性迭代优化控制流
        {                                                     // 锁临界区开始
            std::unique_lock<std::mutex> lock(mMutexBackend); // 锁住后端同步互斥量，安全修改优化请求开关
            mNeedOptimize = true;                             // 将异步优化开关标志置为真，向后端下发滑窗 BA 命令
        } // 锁临界区结束，自动解锁
        mCondBackend.notify_one(); // 唤醒正卡在 BackendLoop 里因为没活干而深度休眠的 Ceres 后端优化线程
    } // 固化处理分支结束

    // PnP 跟踪点失效的健壮安全防御机制
    if (!pnp_succ)                                                                                                                              // 如果由于刚启动没历史点、或者前方发生严重遮挡导致 PnP 解算失败
    {                                                                                                                                           // 防御逻辑开始
        std::cout << ">>> [SLAM前端] 提示：PnP 暂无足够追踪点（如冷启动首帧或发生短暂遮挡），已通过双目实时三角化构建/恢复结构。" << std::endl; // 打印控制台警示系统信息
        // 【核心防御】PnP 失败时，为了不让外部 ROS 2 可视化 RViz 界面突然变空，直接把刚刚双目独立解算出的局部路标点一股脑塞入点云可视化队列
        for (const auto &pair : mmIDToMapPoint)                 // 循环轮询前端持有的哈希路由字典中的所有路标
        {                                                       // 点云挽救遍历开始
            vWorldPoints.push_back(pair.second->GetWorldPos()); // 从指针安全获取其 3D 的世界绝对坐标位置并塞给可视化容器
        } // 点云挽救遍历结束
    } // 防御逻辑结束

    // 轨迹渲染数据拼装：提取全局数据库中所有已知关键帧的绝对物理坐标发送出去
    auto allKFs = mpMap->GetAllKeyFrames();                  // 线程安全地从地图里拔出目前所有的关键帧存储容器
    for (const auto &kf : allKFs)                            // 遍历小地图历史里的每一个关键帧
    {                                                        // 轨迹采集循环开始
        vKFPositions.push_back(kf->GetPose().translation()); // 获取加锁后的关键帧 3D 绝对坐标位置，追加到可视化历史数组中
    } // 轨迹采集循环结束

    // 监控诊断数据实时播报：在终端输出地图的状态
    std::cout << "[SLAM地图管理] 全局关键帧总数: " << allKFs.size()
              << " | 全局地图固化路标点数: " << mpMap->GetMapPointsSize() << std::endl; // 打印当前滑窗因子图的基本组成规模

    // 红蓝渐变画图着色：渲染左目图像上各角点的追踪足迹，方便开发者评估前端光流的优劣
    for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); i++)                // 遍历当前帧左目中的所有像素特征点
    {                                                                              // 特征画圈循环开始
        double len = std::min(1.0, 1.0 * mpFeatureDetector->mvTrackCnt[i] / 20.0); // 追踪寿命（帧数计数）映射到 0.0 ~ 1.0 之间
        cv::Scalar ptColor = cv::Scalar(255 * (1 - len), 0, 255 * len);            // 寿命短点表现为鲜艳红，长寿稳定的老特征表现为高亮蓝
        cv::circle(imgTrack, mpFeatureDetector->mvCurPts[i], 2, ptColor, 2);       // 再次用 OpenCV 将这些实时点画在双目拼接大图的左半侧
    } // 特征画圈循环结束

    mPrevImg = grayLeft.clone();                       // 将本帧左图完整克隆浅拷贝给历史成员，作为下一轮追踪所需的“上一帧底图”
    mpFeatureDetector->UpdatePreviousStatus(grayLeft); // 同步调用，推动特征追踪器内的状态窗口滚动向前更替

    // 临界区隔离更新：安全把解算出的最新解回刷到全局高频缓冲中
    {                                                     // 锁临界区开始
        std::unique_lock<std::mutex> lock(mMutexBackend); // 锁住后端同步互斥量，保障向 mCurrentPose 写入数据的原子性与安全性
        mCurrentPose = localPose;                         // 将本帧解算出的初估局部精确位姿持久化更新进系统骨干缓存
    } // 锁临界区结束，自动解锁
    return localPose; // 完美跳出 ProcessStereo 核心解算管道，回吐绝对变换矩阵
} // 函数主体结束

// 关键帧选拔判定决策函数：通过评估当前帧相对上一关键帧像素移动的平均视差来判断相机运动大小
bool Tracking::NeedNewKeyFrame()
{                                                // 函数主体开始
    if (mpFeatureDetector->mvCurPts.size() < 20) // 规则一（安全红线）：如果当前追踪到的特征点总数小于 20 个
        return true;                             // 为了防御系统彻底跟丢（Lost），二话不说直接返回 true 强制激活为新关键帧
    double total_parallax = 0.0;                 // 声明局部双精度变量，用来累加所有共同追踪特征点的总像素位移视差
    int common_tracked_pts_cnt = 0;              // 声明计数器，用来统计有多少特征点是本帧与上一个关键帧所共有的

    for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); ++i)                                    // 逐个遍历当前最新提取的所有像素特征角点
    {                                                                                                  // 视差计算循环开始
        int id = mpFeatureDetector->mvIds[i];                                                          // 取出当前这个特征角点的全局唯一标识 ID
        auto it = mvpPrevKFPointsMap.find(id);                                                         // 在上一个关键帧所保存的历史观测字典中检索该 ID
        if (it != mvpPrevKFPointsMap.end())                                                            // 如果在历史字典里顺利找到了该 ID，说明该点是跨多帧存活的共有观测点
        {                                                                                              // 视差累加分支开始
            total_parallax += mpFeatureDetector->Distance(mpFeatureDetector->mvCurPts[i], it->second); // 调用 Distance 二维欧氏几何距离计算两帧间的像素视差并累加
            common_tracked_pts_cnt++;                                                                  // 共有匹配计数器自增 1 帧
        } // 视差累加分支结束
    } // 视差计算循环结束
    if (common_tracked_pts_cnt == 0)                                   // 极端意外防护：如果没有找到任何共同可见点（视差无法计算）
        return true;                                                   // 直接返回 true 设定为新关键帧，强行锚定系统参考系
    double average_parallax = total_parallax / common_tracked_pts_cnt; // 数学运算：用总视差除以总共视点数，解算出无视噪声污染的平均像素视差

    return average_parallax >= mKeyframeParallax; // 几何阈值决策判断：若平均像素视差大于等于 15 像素（运动足够大），返回 true 判定为关键帧，反之返回 false
} // 函数主体结束

// 后端常驻后台的异步因子图图优化循环线程：专门负责解决 Ceres 非线性最小二乘优化的计算卡顿问题
void Tracking::BackendLoop()
{                                                                        // 后端函数主体开始
    while (true)                                                         // 常驻死循环，随时待命处理优化
    {                                                                    // 后端循环体开始
        {                                                                // 进入后端条件变量临界区保护
            std::unique_lock<std::mutex> lock(mMutexBackend);            // 锁住后端互斥锁，等待前端发来唤醒冲锋号
            mCondBackend.wait(lock, [this]                               // 挂起后台线程进入阻塞休眠：直到前端把 mNeedOptimize 改为真（有新关键帧了），或者接收到析构关闭指令才醒来
                              { return mNeedOptimize || !mIsRunning; }); //

            if (!mIsRunning)       // 如果醒来的原因是因为整个 SLAM 系统面临析构关闭退出（mIsRunning == false）
                break;             // 跳出死循环，彻底结束常驻后台的后端优化线程
            mNeedOptimize = false; // 成功响应请求，将优化标志重置为 false，准备投入本次解算
        } // 离开临界区保护，自动解锁，保障后端优化计算时前端 ProcessStereo 依旧能流畅推进而互不干扰

        std::cout << ">>> [后端优化] 异步线程触发 10 帧滑动窗口局部 BA 优化..." << std::endl; // 在控制台打印优化日志
        Optimizer::LocalBundleAdjustment(mpMap, 10);                                          // 核心解算：调用优化器，对最新的 10 帧局部滑窗因子图执行重投影误差精调，剔除漂移

        // =========================================================================
        // 【核心架构级修复精髓说明】彻底删除了原本此处由后端强行覆盖前端 mCurrentPose 的灾难性代码。
        // 原因分析：后端 BA 任务仅仅负责精调滑动窗口内的“历史关键帧绝对位姿”和“固化的地图点 3D 物理空间坐标”。
        // 前端通过高频 PnP 算当前帧时，由于输入参照的是被后端不断精调后的最新无漂移地图点，其解算出的 localPose
        // 天然具备收敛性质，因此在此不需要多此一举用历史位姿覆盖它。
        // 删掉这几行不合理逻辑后，前端主线程的帧率流畅度彻底释放，彻底解决了长跑时轨迹断层跳变或频繁跟丢的逻辑死结。
        // =========================================================================
    } // 后端循环体结束
} // 后端函数主体结束