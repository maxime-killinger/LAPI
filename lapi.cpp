/*
** lapi.cpp for LAPI in /home/killin_m/Desktop
**
** Made by Maxime Killinger
** Login   <killin_m@epitech.net>
**
** Started on  Tue Mar  8 13:50:18 2016 Maxime Killinger
** Last update Wed Mar 16 17:25:49 2016 Maxime Killinger
*/

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>
#include <string.h>

using namespace cv;
using namespace std;

int		thresh = 50;
int		N = 11;
const char	*wndname = "LAPI";

static double	angle(Point pt1, Point pt2, Point pt0)
{
  double dx1 = pt1.x - pt0.x;
  double dy1 = pt1.y - pt0.y;
  double dx2 = pt2.x - pt0.x;
  double dy2 = pt2.y - pt0.y;
  return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

static void	drawSquares(Mat& image, const vector<vector<Point> >& squares)
{
  for (size_t i = 0; i < squares.size(); i++)
    {
      const Point* p = &squares[i][0];
      int n = (int)squares[i].size();
      polylines(image, &p, &n, 1, true, Scalar(0,102,51), 3, CV_AA);
    }
  imshow(wndname, image);
}

static void	findSquares( const Mat& image, vector<vector<Point> >& squares )
{
  squares.clear();
  Mat pyr, timg, gray0(image.size(), CV_8U), gray;
  pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
  pyrUp(pyr, timg, image.size());
  vector<vector<Point> > contours;
  for (int c = 0; c < 3; c++)
    {
      int ch[] = {c, 0};
      mixChannels(&timg, 1, &gray0, 1, ch, 1);
      for (int l = 0; l < N; l++)
	{
	  if (l == 0)
	    {
	      Canny(gray0, gray, 0, thresh, 5);
	      dilate(gray, gray, Mat(), Point(-1,-1));
	    }
	  else
	    gray = gray0 >= (l+1)*255/N;
	  findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	  vector<Point> approx;
	  for (size_t i = 0; i < contours.size(); i++)
	    {
	      approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);
	      if (approx.size() == 4 &&
		  fabs(contourArea(Mat(approx))) > 1000 &&
		  isContourConvex(Mat(approx)))
		{
		  double maxCosine = 0;
		  for (int j = 2; j < 5; j++)
		    {
		      double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
		      maxCosine = MAX(maxCosine, cosine);
		    }
		  if (maxCosine < 0.3)
		    squares.push_back(approx);
        }
      }
	}
  }
}

int		main(int ac, char **av)
{
  double	dWidth;
  double	dHeight;
  bool		bSuccess;
  int		width;
  int		height;
  uchar		*raw;

  VideoCapture	cap(0);
  if (!cap.isOpened())
    {
      cout << "Cannot open the video cam" << endl;
      return (-1);
    }
  dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
  dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
  cout << "Frame size : " << dWidth << " x " << dHeight << endl;
  namedWindow("LAPI", CV_WINDOW_AUTOSIZE);
  vector<vector<Point> > squares;
  while (42)
    {
      Mat frame;
      bSuccess = cap.read(frame);
      if (!bSuccess)
	    {
	        cout << "Cannot read a frame from video stream" << endl;
	        break;
	    }
      Mat grey;
      cvtColor(frame, grey, CV_BGR2GRAY);
      width = frame.cols;
      height = frame.rows;
      raw = (uchar *)grey.data;
      imshow(wndname, frame);
      findSquares(frame, squares);
      drawSquares(frame, squares);
      if (waitKey(30) == 27)
	    {
	        cvDestroyAllWindows();
	        cout << "Escape key is pressed by user" << endl;
	        break;
	    }
      if (getWindowProperty("LAPI", 1) == -1)
	    {
            cvDestroyAllWindows();
	        break;
	    }
    }
  return (0);
}
