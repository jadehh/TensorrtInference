#include "include/draw.hpp"

void drawDetect(cv::Mat &frame, const std::vector<Result>& results, const std::vector<std::string> &classNames)
{
    for (const auto &result : results)
    {
        cv::Scalar color = COLORS.at(result.idx % COLORS.size());
        cv::String label = classNames.at(result.idx % classNames.size());

        cv::rectangle(frame, result.box, color, 2);
        cv::putText(frame, label, cv::Point(result.box.x, result.box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
}