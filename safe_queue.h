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

private:
    std::queue<T> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cond;
    bool bShutdown;
};

#endif // SAFE_QUEUE_H