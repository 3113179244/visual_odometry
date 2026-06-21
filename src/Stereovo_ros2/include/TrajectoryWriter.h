#ifndef TRAJECTORYWRITER_H
#define TRAJECTORYWRITER_H

#include <string>
#include <fstream>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

class TrajectoryWriter {
public:
    TrajectoryWriter() = default;
    
    ~TrajectoryWriter() {
        Close();
    }

    // 初始化并创建输出文本
    bool Init(const std::string& filePath) {
        mOutputFile.open(filePath, std::ios::out);
        if (!mOutputFile.is_open()) {
            std::cerr << ">>> [TrajectoryWriter] 错误: 无法打开/创建文件: " << filePath << std::endl;
            return false;
        }
        std::cout << ">>> [TrajectoryWriter] 成功创建评估文件: " << filePath << std::endl;
        return true;
    }

    // 严格按照 KITTI 格式写入当前帧位姿 (输入参数应为前端计算出来的 T_cw)
    void WritePoseKITTI(const Eigen::Isometry3d& T_cw) {
        if (!mOutputFile.is_open()) return;

        // 核心求逆：将世界到相机的 T_cw 转换为相机相对于世界的绝对位姿 T_wc
        Eigen::Isometry3d T_wc = T_cw.inverse();
        Eigen::Matrix3d R = T_wc.linear();
        Eigen::Vector3d t = T_wc.translation();

        // 写入 3x4 齐次矩阵的前三行（共 12 个浮点数，空格分隔）
        mOutputFile << R(0,0) << " " << R(0,1) << " " << R(0,2) << " " << t.x() << " "
                    << R(1,0) << " " << R(1,1) << " " << R(1,2) << " " << t.y() << " "
                    << R(2,0) << " " << R(2,1) << " " << R(2,2) << " " << t.z() << "\n";
    }

    // 手动关闭流
    void Close() {
        if (mOutputFile.is_open()) {
            mOutputFile.close();
            std::cout << ">>> [TrajectoryWriter] 评估文件已安全关闭并保存。" << std::endl;
        }
    }

private:
    std::ofstream mOutputFile;
};

#endif // TRAJECTORYWRITER_H