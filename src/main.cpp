#include <iostream>
#include <fusion/color_constancy.hpp>
#include <fusion/laplacianBlend.hpp>

using namespace std;
using namespace cv;

int mode = 1;
int fps = 30;
int k=0;

/**
 * @brief this function generates the exposedness weight map
 * @param input image 1
 * @param input image 2
 * @return output image is wweight map
 */
Mat generateEMasks(Mat s1,Mat s2){

  // mean and vairance of exposedness mask
  float _gMean = 0.5;
  float _gVariance = 0.25;

  Mat dst;
  Mat src;
  
  s1.copyTo(src);

  src.convertTo(src, CV_32FC3, 1.0, 0);

  cvtColor(src, src, CV_BGR2HSV_FULL);

  src.convertTo(src, CV_32FC3, 1.0/255.0, 0);

  src.copyTo(dst);

  std::vector<cv::Mat_<float> > ichan, ichan1;
  std::vector<cv::Mat_<float> > dchan, dchan1;

  cv::split(src, ichan);

  cv::split(dst, dchan);

  int idx = 2;

  cv::add(ichan[idx], -_gMean,dchan[idx]);

  multiply(dchan[idx], dchan[idx], dchan[idx], 1.0/(2*_gVariance*_gVariance));
  cv::exp(-dchan[idx], dchan[idx]);

  idx = 2;

  cvtColor(src, src, CV_HSV2BGR_FULL);

  dchan[idx].convertTo(dchan[idx], CV_32FC1,1.0,0);

  return dchan[idx];
}

/**
 * @brief function to compute global contrast
 * @param input image is BGR image
 * @return ouput image is a scalar weight map
 */
Mat localContrast(Mat s)
{
  Mat image;
  s.copyTo(image);

  Mat dst;
  cvtColor(image,image,CV_BGR2YCrCb);
  image.convertTo(image,CV_32FC3,1.0/255.0,0);
  std::vector<cv::Mat_<float> > d;
  cv::split(image,d);

  Mat tmp;
  d[0].copyTo(tmp);
  tmp.setTo(cv::Scalar::all(0));
  cv::Laplacian(d[0],d[0],image.depth(),3,1,0);
  cv::absdiff(d[0],tmp,d[0]);
  d[0].convertTo(d[0],CV_32FC1,1.0,0);

  return d[0];
}

/**
 * @brief function to compute the saliency weight map
 * @param input image is BGR image
 * @return  ouput image is scalar weight map
 */
Mat saliency(Mat s)
{

  Mat image;
  s.copyTo(image);
  Mat i, o1, o2, o3;
  cvtColor(image, image, CV_BGR2Lab);
  cv::Scalar mean=cv::mean(image);
  image.copyTo(i);
  cv::medianBlur(i, i, 9);

  i.convertTo(i, CV_32FC3, 1.0/255.0, 0);
  std::vector<cv::Mat_<float> > d;
  cv::split(i, d);
  cv::add(d[0], -mean[0]/255.0, o1);
  cv::multiply(o1, o1, o1, 1.0);
  cv::add(d[1], -mean[1]/255.0, o2);
  cv::multiply(o2, o2, o2, 1.0);
  cv::add(d[2], -mean[2]/255.0, o3);
  cv::multiply(o3, o3, o3, 1.0);
  cv::add(o1, o2, o1);
  cv::add(o1, o3, o3);
  cv::pow(o3, 0.5 ,o1);

  return o1;

}

/**
 * @brief function to calculate the local contrast weight map for image
 * @param image is input image
 * @return output image is weight map
 */
Mat LocalContrast(Mat s)
{

  Mat image;
  s.copyTo(image);
  std::vector<cv::Mat_<float> > d;

  cvtColor(image, image, CV_Lab2BGR);
  image.convertTo(image, CV_32FC3, 1.0/255.0, 0);
  cv::split(image, d);

   Mat blurred, diff;
   double sigma = 7, threshold = 0;
   cv::GaussianBlur(d[0], blurred, Size(), sigma, sigma);

   d[0].convertTo(d[0], CV_32F, 1.0, 0);

   blurred.convertTo(blurred, CV_32F, 1.0, 0);

   cv::absdiff(d[0], blurred, diff);
   // Mat sharpened = d[0] * (1 + amount) + blurred * (-amount);
   return diff;
}


/**
 * @brief this function call the different functions to perform image fusion
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{
  color_correction::contrast_stretching o1;
  color_correction::gray_world o2;

  Mat a, image, frame;

  image = cv::imread("gate_1.jpg", 1);

  clock_t begin = clock();

  a.create(240, 320, CV_8UC(3));
  resize(image, a, a.size(), 0, 0, INTER_NEAREST);

  Mat src1 = o1.run(a);
  Mat src2 = o2.run2(a, 6, 0);

  imshow("original image", a);
  imshow("input 1", src2);

  Mat c11;
  Mat c1 = generateEMasks(a, a);
  c1.convertTo(c11, CV_8U, 255.0, 0);

  cv::Laplacian(c11, c11, c11.depth(), 3, 1, 0);

  {
    // std::cout << "1" << '\n';
    Mat c2 = LocalContrast(src2);
    // std::cout << "2" << '\n';
    Mat c3 = saliency(src1);
    // std::cout << "3" << '\n';
    Mat c4 = saliency(src2);
    // std::cout << "4" << '\n';
    Mat c5 = generateEMasks(src1, src2);
    // std::cout << "5" << '\n';
    Mat c6 = generateEMasks(src2, src1);
    // std::cout << "6" << '\n';
    Mat c7 = localContrast(src1);
    // std::cout << "7" << '\n';
    Mat c8 = localContrast(src2);
    // std::cout << "8" << '\n';

    Mat sum;
    c1.copyTo(sum);

    src1.convertTo(src1, CV_32FC3, 1.0/255.0);
    src2.convertTo(src2, CV_32FC3, 1.0/255.0);
    Mat m1, m2;

    ////////////////////// global contrast ////////////////////////
    {
      Mat result1;
      cv::add(c1, c2, sum);

      cv::divide(c1, sum, m1);
      cv::divide(c2, sum, m2);

      LaplacianBlending lBlend(src1, src2, m1, m2, 4);
      result1 = lBlend.blend();

      result1.convertTo(result1, CV_8U, 255.0, 0);
      imshow("global contrast", result1);
    }
    ////////////////////// saliency ////////////////////////
    {
      Mat result2;
      cv::add(c3, c4, sum);

      cv::divide(c3, sum, m1);
      cv::divide(c4, sum, m2);

      LaplacianBlending lBlend(src1, src2, m1, m2, 4);
      result2 = lBlend.blend();

      result2.convertTo(result2, CV_8U, 255.0, 0);
      imshow("saliency", result2);
    }
    ////////////////////// exposedness ////////////////////////    
    {
      Mat result3;
      cv::add(c5, c6, sum);

      cv::divide(c5, sum, m1);
      cv::divide(c6, sum, m2);

      LaplacianBlending lBlend(src1, src2, m1, m2, 4);
      result3=lBlend.blend();

      result3.convertTo(result3, CV_8U, 255.0, 0);
      imshow("exposedness", result3);
    }
    ////////////////////// local contrast ////////////////////////
    {
      Mat result4;
      cv::add(c7, c8, sum);

      cv::divide(c7, sum, m1);
      cv::divide(c8, sum, m2);

      LaplacianBlending lBlend(src1, src2, m1, m2, 4);
      result4 = lBlend.blend();

      result4.convertTo(result4, CV_8U, 255.0, 0);
      imshow("local contrast", result4);
    }
    cv::add(c1, c2, sum);
    cv::add(sum, c3, sum);
    cv::add(sum, c4, sum);
    cv::add(sum, c5, sum);
    cv::add(sum, c6, sum);
    cv::add(sum, c7, sum);
    cv::add(sum, c8, sum);

    cv::divide(c1, sum, c1);
    cv::divide(c2, sum, c2);
    cv::divide(c3, sum, c3);
    cv::divide(c4, sum, c4);
    cv::divide(c5, sum, c5);
    cv::divide(c6, sum, c6);
    cv::divide(c7, sum, c7);
    cv::divide(c8, sum, c8);

    cvtColor(c1, c1, CV_GRAY2BGR);
    cvtColor(c2, c2, CV_GRAY2BGR);
    cvtColor(c3, c3, CV_GRAY2BGR);
    cvtColor(c4, c4, CV_GRAY2BGR);
    cvtColor(c5, c5, CV_GRAY2BGR);
    cvtColor(c6, c6, CV_GRAY2BGR);
    cvtColor(c7, c7, CV_GRAY2BGR);
    cvtColor(c8, c8, CV_GRAY2BGR);

    cv::multiply(c1, src1, c1);
    cv::multiply(c3, src1, c3);
    cv::multiply(c5, src1, c5);
    cv::multiply(c7, src1, c7);
    cv::multiply(c2, src2, c2);
    cv::multiply(c4, src2, c4);
    cv::multiply(c6, src2, c6);
    cv::multiply(c8, src2, c8);

    cv::add(c1, c2, c1);
    cv::add(c1, c3, c1);
    cv::add(c1, c5, c1);
    cv::add(c1, c7, c1);
    cv::add(c1, c4, c1);
    cv::add(c1, c6, c1);
    cv::add(c1, c8, c1);
    {
      c1.convertTo(c1, CV_8U, 255.0, 0);
      imshow("naive blend", c1);
    }
  }
  cv::waitKey(0);
  return 0;
}
