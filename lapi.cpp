/*
** lapi.cpp for LAPI in /home/killin_m/Desktop
**
** Made by Maxime Killinger
** Login   <killin_m@epitech.net>
**
** Started on  Tue Mar  8 13:50:18 2016 Maxime Killinger
** Last update Thu Mar 17 11:44:32 2016 Maxime Killinger
*/

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/imgproc/imgproc_c.h>

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define abs(x) ((x) > 0 ? (x) : -(x))
#define sign(x) ((x) > 0 ? 1 : -1)

#define STEP_MIN 5
#define STEP_MAX 100

using namespace cv;
using namespace std;

IplImage *image;
CvPoint objectPos = cvPoint(-1, -1);
int h = 0, s = 0, v = 0, tolerance = 10;
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

static void	drawSquares(Mat& frame, const vector<vector<Point> >& squares)
{
  for (size_t i = 0; i < squares.size(); i++)
    {
      const Point* p = &squares[i][0];
      int n = (int)squares[i].size();
      polylines(frame, &p, &n, 1, true, Scalar(0,102,51), 3, CV_AA);
    }
  imshow(wndname, frame);
}

static void	findSquares( const Mat& frame, vector<vector<Point> >& squares )
{
  squares.clear();
  Mat pyr, timg, gray0(frame.size(), CV_8U), gray;
  pyrDown(frame, pyr, Size(frame.cols/2, frame.rows/2));
  pyrUp(pyr, timg, frame.size());
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

CvPoint binarisation(IplImage* frame, int *nbPixels) {
    int x, y;
    CvScalar pixel;
    IplImage *hsv, *mask;
    IplConvKernel *kernel;
    int sommeX = 0, sommeY = 0;
    *nbPixels = 0;
    mask = cvCreateImage(cvGetSize(frame), frame->depth, 1);
    hsv = cvCloneImage(frame);
    cvCvtColor(frame, hsv, CV_BGR2HSV);
    cvInRangeS(hsv, cvScalar(h - tolerance -1, s - tolerance, 0), cvScalar(h + tolerance -1, s + tolerance, 255), mask);
    kernel = cvCreateStructuringElementEx(5, 5, 2, 2, CV_SHAPE_ELLIPSE);
    cvDilate(mask, mask, kernel, 1);
    cvErode(mask, mask, kernel, 1);
    for(x = 0; x < mask->width; x++) {
        for(y = 0; y < mask->height; y++) {
            if(((uchar *)(mask->imageData + y*mask->widthStep))[x] == 255) {
                sommeX += x;
                sommeY += y;
                (*nbPixels)++;
            }
        }
    }
    cvShowImage("Mask", mask);
    cvReleaseStructuringElement(&kernel);
    cvReleaseImage(&mask);
    cvReleaseImage(&hsv);
    if(*nbPixels > 0)
        return cvPoint((int)(sommeX / (*nbPixels)), (int)(sommeY / (*nbPixels)));
    else
        return cvPoint(-1, -1);
}

void addObjectToVideo(IplImage* frame, CvPoint objectNextPos, int nbPixels) {
    int objectNextStepX, objectNextStepY;
    if (nbPixels > 10) {
        if (objectPos.x == -1 || objectPos.y == -1) {
            objectPos.x = objectNextPos.x;
            objectPos.y = objectNextPos.y;
        }
        if (abs(objectPos.x - objectNextPos.x) > STEP_MIN) {
            objectNextStepX = max(STEP_MIN, min(STEP_MAX, abs(objectPos.x - objectNextPos.x) / 2));
            objectPos.x += (-1) * sign(objectPos.x - objectNextPos.x) * objectNextStepX;
        }
        if (abs(objectPos.y - objectNextPos.y) > STEP_MIN) {
            objectNextStepY = max(STEP_MIN, min(STEP_MAX, abs(objectPos.y - objectNextPos.y) / 2));
            objectPos.y += (-1) * sign(objectPos.y - objectNextPos.y) * objectNextStepY;
        }
    } else {
        objectPos.x = -1;
        objectPos.y = -1;
    }
    if (nbPixels > 10)
        cvDrawCircle(frame, objectPos, 15, CV_RGB(255, 0, 0), -1);
    cvShowImage("Color Tracking", frame);
}

void getObjectColor(int event, int x, int y, int flags, void *param = NULL) {
    CvScalar pixel;
    IplImage *hsv;

    if(event == CV_EVENT_LBUTTONUP) {
        hsv = cvCloneImage(image);
        cvCvtColor(image, hsv, CV_BGR2HSV);
        pixel = cvGet2D(hsv, y, x);
        h = (int)pixel.val[0];
        s = (int)pixel.val[1];
        v = (int)pixel.val[2];
        cvReleaseImage(&hsv);

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
  IplImage *hsv;
  int nbPixels;
  CvPoint objectNextPos;

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
//      objectNextPos = binarisation(frame, &nbPixels);
//      addObjectToVideo(frame, objectNextPos, nbPixels);
      cvNamedWindow("Color Tracking", CV_WINDOW_AUTOSIZE);
      cvNamedWindow("Mask", CV_WINDOW_AUTOSIZE);
      cvMoveWindow("Color Tracking", 0, 100);
      cvMoveWindow("Mask", 650, 100);
      cvMoveWindow("LAPI", 325, 580);
//      cvSetMouseCallback("Color Tracking", getObjectColor);
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
