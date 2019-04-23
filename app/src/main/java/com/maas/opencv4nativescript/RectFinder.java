package com.maas.opencv4nativescript;

import android.os.Environment;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.opencv.imgproc.Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C;

/**
 * This class is to find rectangles in the given image.
 */
public class RectFinder {

    /* Logging tag */
    private static final String DEBUG_TAG = "PTransformation";
    /* Set number of times threshold levels to be calculated */
    private static final int thresholdLevelSize = 10;
    /* Set downscale image size  */
    private static final double DOWNSCALE_IMAGE_SIZE = 600f;
    /* Canny threshold value */
    private static final int CANNY_THRESHOLD = 100;

    /**
     * Compare contours by their areas in descending order.
     *
     * @param m1 contourArea1
     * @param m2 contourArea2
     * @return int
     */
    private static Comparator<MatOfPoint2f> AreaDescendingComparator = new Comparator<MatOfPoint2f>() {
        public int compare(MatOfPoint2f m1, MatOfPoint2f m2) {
            double area1 = Imgproc.contourArea(m1);
            double area2 = Imgproc.contourArea(m2);
            return (int) Math.ceil(area2 - area1);
        }
    };
    /* Get external storage directory path */
    private String extStorageDirectory = Environment.getExternalStorageDirectory().toString();
    /* Area low threshold ratio  */
    private double areaLowerThresholdRatio;
    /* Area high threshold */
    private double areaUpperThresholdRatio;
    /* Transformation method (Automatic or Manual)  */
    private String transformationMethod;


    /**
     * RectFinder Constructor
     *
     * @param areaLowerThresholdRatio Area low threshold ratio
     * @param areaUpperThresholdRatio Area high threshold ratio
     * @param method
     */
    public RectFinder(double areaLowerThresholdRatio, double areaUpperThresholdRatio, String method) {
        this.areaLowerThresholdRatio = 0.1D; //areaLowerThresholdRatio;
        this.areaUpperThresholdRatio = 0.85D; //areaUpperThresholdRatio;
        this.transformationMethod = method;
    }

    /**
     * Method to find rectangle
     *
     * @param src Source image in Mat format
     */
    public MatOfPoint2f findRectangle(Mat src) {
        Log.d(DEBUG_TAG, "findRectangle(" + src + ") called");

        // Downscale image for better performance.
        double ratio = DOWNSCALE_IMAGE_SIZE / Math.max(src.width(), src.height());
        Size downscaledSize = new Size(src.width() * ratio, src.height() * ratio);
        Log.d(DEBUG_TAG, "Before downscaling: " + src.size());
        Mat downscaled = new Mat(downscaledSize, src.type());
        Log.d(DEBUG_TAG, "After downscaling: " + downscaled.size());
        // Resize image to downscaledSize.
        Imgproc.resize(src, downscaled, downscaledSize);

//        imwrite(extStorageDirectory + "downscaled.jpg", downscaled);

        // Find rectangles.
        List<MatOfPoint2f> rectangles = findRectangles(downscaled);
        Log.d(DEBUG_TAG, rectangles.size() + " rectangles found.");

        //segregate rectangles by sides
        List<MatOfPoint2f> rectangles4 = new ArrayList<MatOfPoint2f>();
        List<MatOfPoint2f> rectangles5 = new ArrayList<MatOfPoint2f>();
        List<MatOfPoint2f> rectangles6 = new ArrayList<MatOfPoint2f>();
        List<MatOfPoint2f> rectangles7 = new ArrayList<MatOfPoint2f>();
        List<MatOfPoint2f> rectangles8 = new ArrayList<MatOfPoint2f>();
        List<MatOfPoint2f> rectangles9 = new ArrayList<MatOfPoint2f>();
        List<MatOfPoint2f> rectangles10 = new ArrayList<MatOfPoint2f>();

        for (MatOfPoint2f contour : rectangles) {
            if (contour.rows() == 5) {
                rectangles5.add(contour);
            } else if (contour.rows() == 4) {
                rectangles4.add(contour);
            } else if (contour.rows() == 6) {
                rectangles6.add(contour);
            } else if (contour.rows() == 7) {
                rectangles7.add(contour);
            } else if (contour.rows() == 8) {
                rectangles8.add(contour);
            } else if (contour.rows() == 9) {
                rectangles9.add(contour);
            } else if (contour.rows() == 10) {
                rectangles10.add(contour);
            }
        }
        rectangles = rectangles4;

        if (rectangles.size() == 0) {
            if (rectangles5.size() > 0) {
                rectangles = rectangles5;
            } else if (rectangles6.size() > 0) {
                rectangles = rectangles6;
            } else if (rectangles7.size() > 0) {
                rectangles = rectangles7;
            } else if (rectangles8.size() > 0) {
                rectangles = rectangles8;
            } else if (rectangles9.size() > 0) {
                rectangles = rectangles9;
            } else if (rectangles10.size() > 0) {
                rectangles = rectangles10;
            } else {
                return null;
            }
        }

        // Sort the rectangles in descending order.
        Collections.sort(rectangles, AreaDescendingComparator);
        Log.d(DEBUG_TAG, "Sorted rectangles.");

        // Pick up the largest rectangle.
        MatOfPoint2f largestRectangle = rectangles.get(0);
        //  Log.d(DEBUG_TAG, "Before scaling up: " + GeomUtils.pointsToString(largestRectangle));

        // Take back the scale.
        MatOfPoint2f result = GeomUtils.scaleRectangle(largestRectangle, 1f / ratio);
        //  Log.d(DEBUG_TAG, "After scaling up: " + GeomUtils.pointsToString(result));
        //    imwrite( extStorageDirectory+"/largestRecImage.jpg", result);
        return result;
    }

    /**
     * Method to find rectangles in the given source image in Mat format
     *
     * @param src source image in Mat format
     * @return List</                                                                                                                                                                                                                                                               MatOfPoint2f> List of MatOfPoint2f
     */
    public List<MatOfPoint2f> findRectangles(Mat src) {
        Log.d(DEBUG_TAG, "findRectangles(" + src + ") called");
        long timeToExecutefindRectanglesMethodStart = System.nanoTime();

        // Blur the image to filter out the noise.
        Mat blurred = new Mat();
        Imgproc.medianBlur(src, blurred, 11);
//        Imgproc.blur(src, blurred, new Size(3, 3));

        Log.d(DEBUG_TAG, "Blur the image to filter out the noise");

        // Set up images to use.
        Mat gray0 = new Mat(blurred.size(), CvType.CV_8U);
        Mat gray = new Mat();
//        imwrite(extStorageDirectory + "/blurredImage.jpg", blurred);

        // For Core.mixChannels.
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        List<MatOfPoint2f> rectangles = new ArrayList<MatOfPoint2f>();

        List<Mat> sources = new ArrayList<Mat>();
        sources.add(blurred);
        List<Mat> destinations = new ArrayList<Mat>();
        destinations.add(gray0);

        // To filter rectangles by their areas.
        int srcArea = src.rows() * src.cols();
        Log.d(DEBUG_TAG, "Total area size of the image : " + srcArea);

        int threshIteration = 0;
        // AdaptiveThreshold to be done for multiple level of blockSize to get optimal result.
        int[] blockSize = {33, 35, 37, 39, 41, 43, 45};


        // Find squares in every color plane of the image.
        for (int c = 0; c < 1; c++) {
            int[] ch = {c, 0};
            MatOfInt fromTo = new MatOfInt(ch);

            Core.mixChannels(sources, destinations, fromTo);
            Log.d(DEBUG_TAG, "Finding squares in every color plane of the image : " + c);
//            Log.d(DEBUG_TAG, "Applying several threshold levels.");

            // Try several threshold levels.
            for (int bSize : blockSize) {
                if (threshIteration == 0) {

                    // Canny helps to catch squares with gradient shading.
                    Imgproc.Canny(gray0, gray, 0, CANNY_THRESHOLD, 3, true);

                    // Dilate Canny output to remove potential holes between edge segments.
                    Imgproc.dilate(gray, gray, Mat.ones(new Size(3, 3), 0));
                    Log.d(DEBUG_TAG, "Applying threshold  using 'Canny' method. ");
//                    imwrite(extStorageDirectory + "/gradientShading" + c + l + ".jpg", gray);
                } else {
//                    int threshold = (l + 1) * 255 / thresholdLevelSize;
//                    Imgproc.threshold(gray0, gray, threshold, 255, Imgproc.THRESH_BINARY);

//                    Log.d(DEBUG_TAG, "Applying threshold level : " + threshold);

//                    Imgproc.threshold(gray0, grayTemp, 0, 255, Imgproc.THRESH_BINARY);
//                    Imgproc.adaptiveThreshold(gray0, gray, 255.0D, ADAPTIVE_THRESH_MEAN_C, 0, 41, 9.0D);
//                    imwrite(extStorageDirectory + "/thresholdingImage" + c + l + ".jpg", gray);
                    Imgproc.adaptiveThreshold(gray0, gray, 255.0D, ADAPTIVE_THRESH_GAUSSIAN_C, 0, bSize, 9.0D);
//                    imwrite(extStorageDirectory + "/thresholdingImage_" + bSize + threshIteration + ".jpg", gray);
                }
                threshIteration++;
                contours.clear();
                MatOfPoint2f  contour = findContours(contours, gray, srcArea);
                if(contour != null){
                    rectangles.add(contour);
                }
            } // for (int l = 0; l < thresholdLevelSize; l++)
        }

        long timeToExecutefindRectanglesMethod = System.nanoTime() - timeToExecutefindRectanglesMethodStart;
        Log.d(DEBUG_TAG, "Time to execute findRectangles method  (in ms): " + TimeUnit.MILLISECONDS.convert(timeToExecutefindRectanglesMethod, TimeUnit.NANOSECONDS));

        return rectangles;
    }

    /**
     * Find contours
     * @param contours contains contours information
     * @param gray Mat image in gray color
     * @param srcArea size of the source area.
     * @return approximate mat of point2f
     */

    private MatOfPoint2f findContours(List<MatOfPoint> contours, Mat gray, int srcArea) {

        Log.d(DEBUG_TAG, "finding contours...");
        // Find contours and store them all as a list.
        Imgproc.findContours(gray, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

//                Mat hierarchy = new Mat();
//                Mat drawing = Mat.zeros(gray.size(), CV_8UC3);

        Log.d(DEBUG_TAG, "Calling Approximate polygonal curves.");
//                for (MatOfPoint contour : contours) {
//                    MatOfPoint2f contourFloat = GeomUtils.toMatOfPointFloat(contour);
//                    double arcLen = Imgproc.arcLength(contourFloat, true) * 0.02;
//
//                    // Approximate polygonal curves.
//                    MatOfPoint2f approx = new MatOfPoint2f();
//                    Imgproc.approxPolyDP(contourFloat, approx, arcLen, true);
//
//                    if (isRectangle(approx, srcArea)) {
//                        double areaTemp = Math.abs(Imgproc.contourArea(approx));
//                        Log.d(DEBUG_TAG, "Rectangle found. Size is : " + areaTemp);
//                        rectangles.add(approx);
//                    }
////For testing
////                    Random rand = new Random();
////                    Mat hierarchy = new Mat();
////                    Mat drawing = Mat.zeros(gray.size(), CV_8UC3);
////                    for (int i = 0; i < contours.size(); i++) {
////                        int r = rand.nextInt();
////                        int g = rand.nextInt();
////                        int b = rand.nextInt();
////                        Scalar color = new Scalar(r, g, b);
////                        int idx = contours.indexOf(contour);
////                        Imgproc.drawContours(drawing, contours, idx, color, 2, 8, hierarchy, 0, new Point());
////                        imwrite(extStorageDirectory + "/findRectangle.jpg", drawing);
////                        imwrite(extStorageDirectory + "/findRectangle" + c + l + ".jpg", drawing);
////                    }
//// for testing
//                }

        MatOfPoint maxContour = getLargestContour(contours);
        if (maxContour != null) {
            MatOfPoint2f contourFloat = GeomUtils.toMatOfPointFloat(maxContour);

            // Approximate polygonal curves.
            MatOfPoint2f approx = new MatOfPoint2f();
            double arcLen = Imgproc.arcLength(contourFloat, true) * 0.02;
            Imgproc.approxPolyDP(contourFloat, approx, arcLen, true);
//            double arcLen=0;
//            do
//            {
//                arcLen=arcLen+1;
//                Imgproc.approxPolyDP(contourFloat,approx,arcLen,true);
//            }
//            while (approx.rows()>4);

            if (isRectangle(approx, srcArea)) {

                double srcAreaTemp = Math.abs(Imgproc.contourArea(approx));
                Log.d(DEBUG_TAG, "Rectangle found. Area size of the Rectangle : " + srcAreaTemp);

                return approx;

            }
        }
        return null;
    }

    /**
     * To get largest contour
     *
     * @param contours
     * @return MatOfPoint largest contour
     */
    private MatOfPoint getLargestContour(List<MatOfPoint> contours) {
        double maxArea = -1;
        MatOfPoint maxContour = null;
        for (MatOfPoint contour : contours) {

            //  double arcLen = Imgproc.arcLength(contourFloat, true) * 0.02;
            double contourArea = Math.abs(Imgproc.contourArea(contour));
            if (contourArea > maxArea) {
                maxArea = contourArea;
                maxContour = contour;
            }
        }
        if (maxContour != null) {
            return maxContour;
        } else {
            return null;
        }

    }

    /**
     * Private method to check the polygon is rectangle or not
     *
     * @param polygon MatOfPoint2f
     * @param srcArea integer source area
     * @return boolean true/false
     */
    private boolean isRectangle(MatOfPoint2f polygon, int srcArea) {
        MatOfPoint polygonInt = GeomUtils.toMatOfPointInt(polygon);

        // Check polygon has 4 sides or not
        if (polygon.rows() == 4) {
            return isRectangleWithinSourceArea(polygon, srcArea);
        }
        if (polygon.rows() == 5) {
            return isRectangleWithinSourceArea(polygon, srcArea);
        }
        if (polygon.rows() == 6) {
            return isRectangleWithinSourceArea(polygon, srcArea);
        }
        if (polygon.rows() == 7) {
            return isRectangleWithinSourceArea(polygon, srcArea);
        }
        if (polygon.rows() == 8) {
            return isRectangleWithinSourceArea(polygon, srcArea);
        }
        if (polygon.rows() == 9) {
            return isRectangleWithinSourceArea(polygon, srcArea);
        }
        if (polygon.rows() == 10) {
            return isRectangleWithinSourceArea(polygon, srcArea);
        }

//
//        if (!Imgproc.isContourConvex(polygonInt)) {
//            return false;
//        }
        return false;
//        // Check if the all angles are more than 72.54 degrees (cos 0.3).
//        double maxCosine = 0;
//        Point[] approxPoints = polygon.toArray();
//
//        for (int i = 2; i < 5; i++) {
//            double cosine = Math.abs(GeomUtils.angle(approxPoints[i % 4], approxPoints[i - 2], approxPoints[i - 1]));
//            maxCosine = Math.max(cosine, maxCosine);
//        }
//
//        return !(maxCosine >= 0.3);
    }

    /**
     * To check the appropriate rectangle size is close to the object size.
     *
     * @param polygon
     * @param srcArea
     * @return true/false
     */
    private boolean isRectangleWithinSourceArea(MatOfPoint2f polygon, int srcArea) {
        double area = Math.abs(Imgproc.contourArea(polygon));
        if (area < srcArea * areaLowerThresholdRatio || area > srcArea * areaUpperThresholdRatio) {
            return false;
        }
        return true;
    }


}
