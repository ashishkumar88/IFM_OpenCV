//============================================================================
// Name        : IFM_OpenCV.cpp
// Author      : Ashish Kumar
// Version     : 1.0v
// Copyright   : Your copyright notice
// Description : A c++ program which performs printed label detection and also
//               finds the position, orientation, area and perimeter of the
//               printed label as well as confidence of label detection.
//============================================================================

#include <iostream>
#include <opencv2/opencv.hpp>
#include<stdio.h>

using namespace cv;
using namespace std;

/******************************************************************************
* A class to store position of an object
******************************************************************************/
class Position {
private:
	double width;
	double height;
public:
	double getWidth() {
		return width;
	}

	double getHeight() {
		return height;
	}

	Position(double w = 0.0, double h = 0.0) {
		width = w;
		height = h;
	}

};

/******************************************************************************
* A class to store details of a printed label
******************************************************************************/
class PrintedLabel {
private: 
	Position position;
	double orientation;
	double area;
	double perimeter;
	double confidence;

public:	
	Position getPosition() {
		return position;
	}

	double getOrientation() {
		return orientation;
	}

	double getArea() {
		return area;
	}

	double getPerimeter() {
		return perimeter;
	}

	double getConfidence() {
		return confidence;
	}
};

/******************************************************************************
* A class for the detection of printed labels. This class uses the opencv lib-
* rary for the printed label dection and has a function to print details of t-
* he labels.
******************************************************************************/
class PrintedLabelDetector {
private:
	char response;
	Mat image;
	list<PrintedLabel> printedLabels;

public:

	/**************************************************************************
	* A member function which loads image from the hard drive or from a ROS to-
	* pic
	**************************************************************************/
	void loadImage() {
		cout << "Enter (f) for a file and (t) for a ROS topic: ";
		cin >> response;
		if(response == 'f') {
			image = loadImageFromFile();
		} else if (response == 't') {
			image = loadImageFromROSTopic();
		} else {
			cout << "Response is not recognized. Program will be terminated." << endl;
		}
	}

	/**************************************************************************
	* A member function which loads image from the hard drive 
	**************************************************************************/
	Mat loadImageFromFile() {
		string fileName = "";
		while(true) {
			cout << "Enter file name : ";
			// Test if file was read successfully
			if (!std::getline(cin, fileName)) { 
				cout << "I/O Error has occurred";
			}

			// Test if the file exits and the file name is not empty
			if(!fileName.empty()) {
				ifstream f(fileName.c_str());
				if(f.good()) {
					break;
				} else {
					cout << "The file name entered is incorrect.";
				}
			} else {
				cout << "The file name entered is empty.";
			}		
			cout << endl;
		}
		Mat image = imread(fileName, CV_LOAD_IMAGE_COLOR); ;
	
		// Test if the file is supported by the open cv library
		if(!image.data) {
			cout <<  "This file is not supported" << endl ;
		} 
		
		return image;
	}
	
	/**************************************************************************
	* A member function which loads image from a ROS topic
	**************************************************************************/
	Mat loadImageFromROSTopic() {
		Mat image;
		return image;
	}

	/**************************************************************************
	* This function uses opencv apis for the labels detection. The algorithm is
	* as follows:
	* 1) Apply the sharpening filter on the image.
	* 2) Detect all the rectangles in the image.
	* 3) For a single rectangle, perform a histogram check of its color distri-
	*    to ensure that it lies in the range of white and black.
	* 4) Repeat step (3) for each of the rectangles.
	**************************************************************************/
	void processImage() {
		cout << "Started processing" << endl;
		sharpenImage(image);
		showImage(image);
		return;
	}

	/**************************************************************************
	* Used a sharpening filter on the original image.
	* Source : https://en.wikipedia.org/wiki/Kernel_(image_processing)
	**************************************************************************/
	void sharpenImage(Mat &image) {	
		Mat img_higher_contrast;
		image.convertTo(img_higher_contrast, -1, 1.1, 0); 
		
		Mat sharpened;
		Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
		filter2D(img_higher_contrast, sharpened, img_higher_contrast.depth(), kernel);

		kernel = (Mat_<float>(5, 5) << -1/8.0,-1/8.0,-1/8.0,-1/8.0,-1/8.0, 
		-1/8.0,2/8.0,2/8.0,2/8.0,-1/8.0, 
		-1/8.0,2/8.0,8/8.0,2/8.0,-1/8.0, 
		-1/8.0,2/8.0,2/8.0,2/8.0,-1/8.0, 
		-1/8.0,-1/8.0,-1/8.0,-1/8.0,-1/8.0);
		filter2D(sharpened, image, sharpened.depth(), kernel);
		
		vector<vector<Point> > squares;
		findSquares(image, squares);

		vector<vector<Point> >::iterator it = squares.begin();
		for(; it != squares.end(); ++it) {
			vector<Point> points = *it;
			Mat square = Mat(points.size(),2,CV_64F,points.data());
		}
        drawSquares(image, squares);
	}

	void showImage(Mat &image) {
		namedWindow("Image", WINDOW_AUTOSIZE);
		imshow("Image", image); 	
		waitKey(0); 
	}

	void publishResults() {
		cout << "The results are as follows: " << endl;
		list<PrintedLabel>::iterator it;
		int counter = 1;
		for (it = printedLabels.begin(); it != printedLabels.end(); ++it){
			cout << counter << ")";
			cout << "\t" << "(" << it->getPosition().getWidth() << ", " << it->getPosition().getHeight() << ")";
			cout << "\t" << it->getOrientation();
			cout << "\t" << it->getArea();
			cout << "\t" << it->getPerimeter();
			cout << "\t" << it->getConfidence();
		}
	}

	double angle( Point pt1, Point pt2, Point pt0 ) {
		double dx1 = pt1.x - pt0.x;
		double dy1 = pt1.y - pt0.y;
		double dx2 = pt2.x - pt0.x;
		double dy2 = pt2.y - pt0.y;
		return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
	}

	/**************************************************************************
	* returns sequence of squares detected on the image.
	* the sequence is stored in the specified memory storage
	* Code taken from open cv samples
	**************************************************************************/
	void findSquares( const Mat& image, vector<vector<Point> >& squares ) {

		const int thresh = 50, N = 11;
		squares.clear();

		Mat pyr, timg, gray0(image.size(), CV_8U), gray;

		// down-scale and upscale the image to filter out the noise
		pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
		pyrUp(pyr, timg, image.size());
		vector<vector<Point> > contours;

		// find squares in every color plane of the image
		for( int c = 0; c < 3; c++ )
		{
			int ch[] = {c, 0};
			mixChannels(&timg, 1, &gray0, 1, ch, 1);

			// try several threshold levels
			for( int l = 0; l < N; l++ )
			{
				// hack: use Canny instead of zero threshold level.
				// Canny helps to catch squares with gradient shading
				if( l == 0 )
				{
					// apply Canny. Take the upper threshold from slider
					// and set the lower to 0 (which forces edges merging)
					Canny(gray0, gray, 0, thresh, 5);
					// dilate canny output to remove potential
					// holes between edge segments
					dilate(gray, gray, Mat(), Point(-1,-1));
				}
				else
				{
					// apply threshold if l!=0:
					//     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
					gray = gray0 >= (l+1)*255/N;
				}

				// find contours and store them all as a list
				findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

				vector<Point> approx;

				// test each contour
				for( size_t i = 0; i < contours.size(); i++ )
				{
					// approximate contour with accuracy proportional
					// to the contour perimeter
					approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

					// square contours should have 4 vertices after approximation
					// relatively large area (to filter out noisy contours)
					// and be convex.
					// Note: absolute value of an area is used because
					// area may be positive or negative - in accordance with the
					// contour orientation
					if( approx.size() == 4 &&
						fabs(contourArea(Mat(approx))) > 1000 &&
						isContourConvex(Mat(approx)) )
					{
						double maxCosine = 0;

						for( int j = 2; j < 5; j++ )
						{
							// find the maximum cosine of the angle between joint edges
							double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
							maxCosine = MAX(maxCosine, cosine);
						}

						// if cosines of all angles are small
						// (all angles are ~90 degree) then write quandrange
						// vertices to resultant sequence
						if( maxCosine < 0.3 )
							squares.push_back(approx);
					}
				}
			}
		}
	}


	/**************************************************************************
	* the function draws all the squares in the image
	* Code taken from open cv samples
	**************************************************************************/
	void drawSquares( Mat& image, const vector<vector<Point> >& squares ) {
		string wndname = "Square Detection Demo";
		for( size_t i = 0; i < squares.size(); i++ )
		{
			const Point* p = &squares[i][0];
			int n = (int)squares[i].size();
			polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, LINE_AA);
		}

		imshow(wndname, image);
	}

};

int main() {
	PrintedLabelDetector pld;
	pld.loadImage();
	pld.processImage();
	pld.publishResults();
	return 0;
}
