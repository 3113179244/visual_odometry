#ifndef SAFE_QUEUE_H
#define SAFE_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>

template <typename T>
class SafeQueue
{
public:
    SafeQueue() : bShutdown(false) {}

    // 前端调用：向队列压入数据
    void push(const T &item)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_queue.push(item);
        m_cond.notify_one(); // 唤醒正在等待的后端线程
    }

    // 后端调用：阻塞等待提取数据
    bool pop(T &item)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        // 如果队列为空且系统未要求关闭，则一直阻塞等待
        m_cond.wait(lock, [this]()
                    { return !m_queue.empty() || bShutdown; });

        if (m_queue.empty() || bShutdown)
        {
            return false;
        }

        item = m_queue.front();
        m_queue.pop();
        return true;
    }

    // 清空队列
    void clear()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        std::queue<T> empty_queue;
        std::swap(m_queue, empty_queue);
    }

    // 系统退出时调用，唤醒所有阻塞的线程
    void shutdown()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        bShutdown = true;
        m_cond.notify_all();
    }

    // ====== DEBUG REFACTOR CODE START ======
    // 💡 新增：线程安全地获取当前队列中积压的元素数量
    size_t size()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_queue.size();
    }

    // 💡 新增：线程安全地判断当前队列是否为空
    bool empty()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_queue.empty();
    }
    // ====== DEBUG REFACTOR CODE END ======

private:
    std::queue<T> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cond;
    bool bShutdown;
};

#endif // SAFE_QUEUE_H