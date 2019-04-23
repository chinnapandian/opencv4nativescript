package com.maas.opencv4nativescript;

import android.annotation.SuppressLint;
import android.os.Environment;
import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Class to perform perspective transformation for given image
 */
public class PerspectiveTransformation {
    private static final String DEBUG_TAG = "PTransformation";
    /**
     * Comparator to compare pointX
     *
     * @return sorted list in ascending order.
     */
    private static Comparator<Point> PointXAscendingComparator = new Comparator<Point>() {
        public int compare(Point p1, Point p2) {
            double pointX1 = p1.x;
            double pointX2 = p2.x;
            return (int) Math.ceil(pointX1 - pointX2);
        }
    };
    String extStorageDirectory = Environment.getExternalStorageDirectory().toString();

    /**
     * Constructor
     */
    public PerspectiveTransformation() {
    }

    /**
     * Method to transform the source image based on the selected rectangle corners.
     *
     * @param src     source image in Mat format
     * @param corners List of rectangle corners in MatOfPoint2f format
     * @return Mat a transformed image
     */
    public Mat transform(Mat src, MatOfPoint2f corners) {
        //sorting obtained rectangle corners
        MatOfPoint2f sortedCorners = sortCorners(corners);
//        MatOfPoint2f sortedCorners = sortPointsByComparator(corners);

        // Get the rectangle size with sorted corners
        Size size = getRectangleSize(sortedCorners);

        Log.d(DEBUG_TAG, String.format("Transforming to: %f %f", size.width, size.height));

        Mat result = Mat.zeros(size, src.type());
        MatOfPoint2f imageOutline = getOutline(result);

        //Getting transformation matrix
        Mat transformation = Imgproc.getPerspectiveTransform(sortedCorners, imageOutline);
//        Imgcodecs.imwrite(extStorageDirectory+"/tranformationImage.jpg",transformation);

        // Applying the transformation matrix on the source image to perspective correction
        Imgproc.warpPerspective(src, result, transformation, size);
//        Imgcodecs.imwrite(extStorageDirectory+"/wrapedImage.jpg",result);

        return result;
    }

    /**
     * Private method to get rectangle size.
     *
     * @param rectangle list of sorted corner points
     * @return size the size of the rectangle
     */
    private Size getRectangleSize(MatOfPoint2f rectangle) {
        Point[] corners = rectangle.toArray();

        double top = getDistance(corners[0], corners[1]);
        double right = getDistance(corners[1], corners[2]);
        double bottom = getDistance(corners[2], corners[3]);
        double left = getDistance(corners[3], corners[0]);

        double averageWidth = (top + bottom) / 2f;
        double averageHeight = (right + left) / 2f;

        return new Size(new Point(averageWidth, averageHeight));
    }

    /**
     * Private method to get distance between two points
     *
     * @param p1 points1
     * @param p2 point2
     * @return double the distance of two points
     */
    private double getDistance(Point p1, Point p2) {
        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Private method to get outline of the image
     *
     * @param image the image to be got outline
     * @return MatOfPoint2f the outline image in Mat
     */
    private MatOfPoint2f getOutline(Mat image) {
        Point topLeft = new Point(0, 0);
        Point topRight = new Point(image.cols(), 0);
        Point bottomRight = new Point(image.cols(), image.rows());
        Point bottomLeft = new Point(0, image.rows());
        Point[] points = {topLeft, topRight, bottomRight, bottomLeft};

        MatOfPoint2f result = new MatOfPoint2f();
        result.fromArray(points);

        return result;
    }

    private MatOfPoint2f sortPointsByComparator(MatOfPoint2f points) {
        List<Point> pointsList = points.toList();
        Collections.sort(pointsList, new Comparator<Point>() {

            public int compare(Point o1, Point o2) {
                return Double.compare(o1.x, o2.x);
            }
        });
        MatOfPoint2f result = new MatOfPoint2f();
        result.fromList(pointsList);
        return result;
    }

    /**
     * Private method to sort rectangle corners
     *
     * @param corners List of rectangle corner points
     * @return MatOfPoint2f sortted corners in Mat
     */
    @SuppressLint("LongLogTag")
    public MatOfPoint2f sortCorners(MatOfPoint2f corners) {

        // Get center point of the rectangle
        Point center = getMassCenter(corners);
        List<Point> points = corners.toList();
        List<Point> topPoints = new ArrayList<Point>();
        List<Point> bottomPoints = new ArrayList<Point>();

        //Segregate rectangle point into two: topPoints and bottomPoints
        for (Point point : points) {
            if (point.y < center.y) {
                topPoints.add(point);
            } else {
                bottomPoints.add(point);
            }
        }
        MatOfPoint2f result = new MatOfPoint2f();

        Collections.sort(topPoints, PointXAscendingComparator);
        Collections.sort(bottomPoints, PointXAscendingComparator);


        if (topPoints.size() >= 2 && bottomPoints.size() >= 2) {
            //Get top left point
            Point topLeft = topPoints.get(0);
            // Point topLeft = topPoints.get(0).x > topPoints.get(1).x ? topPoints.get(1) : topPoints.get(0);
            //Get top right point
            Point topRight = topPoints.get(topPoints.size() - 1);
//        Point topRight = topPoints.get(0).x > topPoints.get(1).x ? topPoints.get(0) : topPoints.get(1);
            //Get bottom left point
            Point bottomLeft = bottomPoints.get(0);
//        Point bottomLeft = bottomPoints.get(0).x > bottomPoints.get(1).x ? bottomPoints.get(1) : bottomPoints.get(0);
            //Get bottom right point
            Point bottomRight = bottomPoints.get(bottomPoints.size() - 1);
//        Point bottomRight = bottomPoints.get(0).x > bottomPoints.get(1).x ? bottomPoints.get(0) : bottomPoints.get(1);

//        Log.d(DEBUG_TAG, "Sorted corners:");
//        Log.d(DEBUG_TAG, String.format("      top left: %f %f", topLeft.x, topLeft.y));
//        Log.d(DEBUG_TAG, String.format("     top right: %f %f", topRight.x, topRight.y));
//        Log.d(DEBUG_TAG, String.format("   bottom left: %f %f", bottomLeft.x, bottomLeft.y));
//        Log.d(DEBUG_TAG, String.format("  bottom right: %f %f", bottomRight.x, bottomRight.y));

            //since rectangle corner is curve, the points are being added by 10
            // to cover the edges of the image.
            topLeft.x = topLeft.x - 10;
            topLeft.y = topLeft.y - 10;
            topRight.x = topRight.x + 10;
            topRight.y = topRight.y - 10;
            bottomRight.x = bottomRight.x + 10;
            bottomRight.y = bottomRight.y + 10;
            bottomLeft.x = bottomLeft.x - 10;
            bottomLeft.y = bottomLeft.y + 10;

            // sort the points in clock-wise order
            Point[] sortedPoints = {topLeft, topRight, bottomRight, bottomLeft};
            result.fromArray(sortedPoints);
        }

        return result;
    }

    /**
     * Private method to get center point of the rectangle
     *
     * @param points list of rectangle points
     * @return Point center point of the rectangle
     */
    private Point getMassCenter(MatOfPoint2f points) {
        double xSum = 0;
        double ySum = 0;
        List<Point> pointList = points.toList();
        int len = pointList.size();
        for (Point point : pointList) {
            xSum += point.x;
            ySum += point.y;
        }
        return new Point(xSum / len, ySum / len);
    }

}
