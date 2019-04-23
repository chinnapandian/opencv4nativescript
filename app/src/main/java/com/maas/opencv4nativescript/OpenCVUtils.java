package com.maas.opencv4nativescript;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Environment;
import android.util.Log;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import javax.activation.DataHandler;
import javax.activation.DataSource;
import javax.activation.FileDataSource;
import javax.mail.Authenticator;
import javax.mail.BodyPart;
import javax.mail.Message;
import javax.mail.MessagingException;
import javax.mail.Multipart;
import javax.mail.PasswordAuthentication;
import javax.mail.Session;
import javax.mail.Transport;
import javax.mail.internet.InternetAddress;
import javax.mail.internet.MimeBodyPart;
import javax.mail.internet.MimeMessage;
import javax.mail.internet.MimeMultipart;

import static android.content.ContentValues.TAG;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import static org.opencv.imgproc.Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY;
import static org.opencv.imgproc.Imgproc.circle;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.line;
import static org.opencv.imgproc.Imgproc.rectangle;

/**
 * OpenCV utility class which contains OpenCV related operations.
 */
public class OpenCVUtils {

    private static final String DEBUG_TAG = "PTransformation";
    private static Context context = null;
    private static Session session = null;
    private static String rec, subject, textMessage, attachmentFile, attachmentFileSrc, sendMailStatus, logFile;
    //    private static Mat wrappedImage = null;
    private static String extStoragePublicDirectory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM).toString();
    /**
     * OpenCV loader to initialize OpenCV modules.
     */
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(context) {

        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.d(TAG, "OpenCV loaded successfully");
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    /**
     * Constructor
     *
     * @param appContext Application context
     */
    public OpenCVUtils(Context appContext) {
        Log.d(DEBUG_TAG, "Calling OpenCVUtils");
        context = appContext;
//        wrappedImage = new Mat();

        // Initializing OpenCV modules
        if (!OpenCVLoader.initDebug()) {
            Log.d(DEBUG_TAG, "Unable to load OpenCV");
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    /**
     * Static method to create logs
     *
     * @param fileName
     */
    public static void createLogs(String fileName) {
        try {
            String extStorageDirectory = Environment.getExternalStorageDirectory().toString();
            logFile = extStorageDirectory + "/Logcat" + fileName + ".txt";
            File fdelete = new File(logFile);
            if (fdelete.exists()) {
                if (fdelete.delete()) {
                    Log.d(DEBUG_TAG, "createLogs: file Deleted :" + logFile);
                } else {
                    Log.d(DEBUG_TAG, "createLogs: file not deleted :" + logFile);
                }
            }
            Runtime.getRuntime().exec("logcat -c");
            // Runtime.getRuntime().exec("logcat -f " + logFile + " PTransformation:D *:S");
            Runtime.getRuntime().exec("logcat -f " + logFile);
        } catch (IOException e) {
            Log.d(DEBUG_TAG, e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Method to delete logs
     *
     * @param fileName
     */
    public static void deleteLogs(String fileName) {
        String extStorageDirectory = Environment.getExternalStorageDirectory().toString();
        String logFileName = extStorageDirectory + "/Logcat" + fileName + ".txt";
        File fdelete = new File(logFileName);
        if (fdelete.exists()) {
            if (fdelete.delete()) {
                Log.d(DEBUG_TAG, "deleteLogs: file Deleted :" + fileName);
            } else {
                Log.d(DEBUG_TAG, "deleteLogs: file not deleted :" + fileName);
            }
        }
    }

    public static Mat performPerspectiveCorrection(String imgURI) {


        Log.d(DEBUG_TAG, "performPerspectiveCorrection(" + imgURI + ") called");
        // Get file name from the given URI
        String fileName = getFileName(imgURI);

        // Initialize logs
        createLogs(fileName);

        // Create an OpenCV mat from the bitmap.
        Mat srcMat = Imgcodecs.imread(imgURI, CV_8UC1);
        String imgURISrc = extStoragePublicDirectory + "/" + fileName + ".jpg";

        //Write to local external storage.
        //imwrite(imgURISrc,srcMat);
        long timeToFindRectangleStart = System.nanoTime();

        // Find the largest rectangle.
        // Find image views.
        RectFinder rectFinder = new RectFinder(0.2, 0.98, "A");
        MatOfPoint2f rectangle = rectFinder.findRectangle(srcMat);

        long timeToFindRectangle = System.nanoTime() - timeToFindRectangleStart;
        Log.d(DEBUG_TAG, "Time to find rectangle (in ms) :" + TimeUnit.MILLISECONDS.convert(timeToFindRectangle, TimeUnit.NANOSECONDS));

        Mat wrappedImage = new Mat();

        if (rectangle == null) {
            //no rectangle found
            wrappedImage = srcMat;
            Log.d(DEBUG_TAG, "Rectangle found.");
            //return "";
        } else {
            Log.d(DEBUG_TAG, "Performing Perspective Correction...");
            long timeToTransformRectangleStart = System.nanoTime();

            // Transform the rectangle.
            PerspectiveTransformation perspective = new PerspectiveTransformation();
            wrappedImage = perspective.transform(srcMat, rectangle);
            long timeToTransformRectangle = System.nanoTime() - timeToTransformRectangleStart;
            Log.d(DEBUG_TAG, "Time to transform rectangle (in ms) :" + TimeUnit.MILLISECONDS.convert(timeToTransformRectangle, TimeUnit.NANOSECONDS));
        }


        return wrappedImage;
    }

    public static String drawShape(String imgURI, String pointsStr, String imageActualSize, int pointsCounter) {
        // Get file name from the given URI
        String fileName = getFileName(imgURI);

        // Create an OpenCV mat from the bitmap.
//        Mat srcMat = Imgcodecs.imread(imgURI, CV_8UC1);

        Mat srcMat = Imgcodecs.imread(imgURI);
        String[] points = pointsStr.split("-");
        String[] imgSize = imageActualSize.split("-");
        int width = srcMat.cols();
        int height = srcMat.rows();
        Point pointCenter = new Point((Double.parseDouble(points[0]) / (Double.parseDouble(imgSize[0])) * width), (Double.parseDouble(points[1]) / (Double.parseDouble(imgSize[1])) * height));
        circle(srcMat, pointCenter, 77, new Scalar(0, 255, 0), 5);
        String fileExt = imgURI.substring(imgURI.indexOf("."));
        Integer p1 = Double.valueOf(points[0]).intValue();
//        Integer p2 = Double.valueOf(points[1]).intValue();
        String imgURITemp = imgURI.substring(0, imgURI.indexOf(".")) + "_TEMP" + pointsCounter + fileExt;
        imwrite(imgURITemp, srcMat);
        return imgURITemp;
    }

    /**
     * Perform perspective correction manually.
     *
     * @param imgURI
     * @param rectanglePointsStr
     * @param imageActualSize
     * @return transformed image URI
     */
    public static String performPerspectiveCorrectionManual(String imgURI, String rectanglePointsStr, String imageActualSize) {

        String fileName = getFileName(imgURI);
//         Initialize logs
        createLogs(fileName);
        //Read original image
        //  String imgURIOrg = imgURI.substring(0, imgURI.indexOf("_TEMP")) + imgURI.substring(imgURI.indexOf("."));
        Mat srcMat = Imgcodecs.imread(imgURI);
        int width = srcMat.cols();
        int height = srcMat.rows();

        String[] rectanglPoints = rectanglePointsStr.split("#");
        String[] imgSize = imageActualSize.split("-");

        List<Point> recPoints = new ArrayList<Point>();
        MatOfPoint2f matOfPoints2f = new MatOfPoint2f();
        for (String pointStr : rectanglPoints
                ) {
            String[] point = pointStr.split("-");

            Point pointer = new Point((Double.parseDouble(point[0]) / (Double.parseDouble(imgSize[0])) * width), (Double.parseDouble(point[1]) / (Double.parseDouble(imgSize[1])) * height));
            recPoints.add(pointer);
//            recPoints.add(new Point(Double.parseDouble(point[0]) * width,Double.parseDouble(point[1]) * height));

//            matOfPoints2f.fromArray(matRecPoints);
//
        }
        Mat matRecPoints = Converters.vector_Point2f_to_Mat(recPoints); //vector_Point2f_to_Mat(quad_pts);

        matOfPoints2f = new MatOfPoint2f(matRecPoints);


        // Transform the rectangle.
        PerspectiveTransformation perspective = new PerspectiveTransformation();
        Mat wrappedImage = perspective.transform(srcMat, matOfPoints2f);
        Mat srcMatGray = new Mat();
        cvtColor(wrappedImage, srcMatGray, COLOR_BGR2GRAY);
//        Imgproc.adaptiveThreshold(thr, thr, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 41,9);
//
//        String extStorageDirectoryTemp = extStoragePublicDirectory + "/" + fileName + "_";
//        String fileNamePT = extStorageDirectoryTemp+"transformed.jpg";
//        imwrite(fileNamePT, thr);

        String fileNamePT = performAdaptiveThreshold(srcMatGray, fileName, 41, null);
        return fileNamePT;

    }

    /**
     * Method to perform perspective correction
     *
     * @param imgURI image URI location
     * @param method method(Automatic - A or Manual - M) in which perform perspective correction
     * @return String A transformed image
     */
    public static String performPerspectiveCorrection(String imgURI, String method, String transformedFilePath) {


        Log.d(DEBUG_TAG, "performPerspectiveCorrection(" + imgURI + method + ") called");
        // Get file name from the given URI
        String fileName = getFileName(imgURI);

        // Initialize logs
        createLogs(fileName);

        // Create an OpenCV mat from the bitmap.
        Mat srcMat = Imgcodecs.imread(imgURI, CV_8UC1);
        String imgURISrc = extStoragePublicDirectory + "/" + fileName + ".jpg";

        int width = srcMat.cols();
        int height = srcMat.rows();

        //Write to local external storage.
        //imwrite(imgURISrc,srcMat);
        long timeToFindRectangleStart = System.nanoTime();

        // Find the largest rectangle.
        // Find image views.
        RectFinder rectFinder = new RectFinder(0.2, 0.98, method);
        MatOfPoint2f rectangle = rectFinder.findRectangle(srcMat);

        long timeToFindRectangle = System.nanoTime() - timeToFindRectangleStart;
        Log.d(DEBUG_TAG, "Time to find rectangle (in ms) :" + TimeUnit.MILLISECONDS.convert(timeToFindRectangle, TimeUnit.NANOSECONDS));

        Mat wrappedImage = new Mat();
        String pointStr = "RPTSTR#";

        if (rectangle == null) {
            //no rectangle found
            wrappedImage = srcMat;
            Log.d(DEBUG_TAG, "Rectangle found.");
            //return "";
        } else {
            Log.d(DEBUG_TAG, "Performing Perspective Correction...");
            long timeToTransformRectangleStart = System.nanoTime();

            // Transform the rectangle.
            PerspectiveTransformation perspective = new PerspectiveTransformation();
            wrappedImage = perspective.transform(srcMat, rectangle);

            MatOfPoint2f sortedRecpoints = perspective.sortCorners(rectangle);
            //Get captured rectangle points for GUI
            for (Point point : sortedRecpoints.toArray()
                    ) {
                Double pointX = point.x;
                if (pointX <= 0) {
                    pointX = 1.0;
                }
                Double pointY = point.y;
                if (pointY <= 0) {
                    pointY = 1.0;
                }
                pointStr += Double.valueOf(pointX / width) + "%" + Double.valueOf(pointY / height) + "#";
            }

            // Draw contour on original image.
            drawLinesOnImage(sortedRecpoints, imgURI, extStoragePublicDirectory, fileName);

            long timeToTransformRectangle = System.nanoTime() - timeToTransformRectangleStart;
            Log.d(DEBUG_TAG, "Time to transform rectangle (in ms) :" + TimeUnit.MILLISECONDS.convert(timeToTransformRectangle, TimeUnit.NANOSECONDS));
        }


        //Apply threshold to transformed image and store it in local storage.

////        long timeToConvertImageToGrayStart = System.nanoTime();
//        Mat thr = new Mat();
//       // cvtColor(dstMat, thr, COLOR_BGR2GRAY);
////        long timeToConvertImageToGray = System.nanoTime() - timeToConvertImageToGrayStart;
//
//        Log.d(DEBUG_TAG, "Time to Converting image into gray (in ms): " + TimeUnit.MILLISECONDS.convert(timeToConvertImageToGray, TimeUnit.NANOSECONDS));
//        long timeToApplyAdaptiveThresholdStart = System.nanoTime();
//        Imgproc.adaptiveThreshold(wrappedImage, thr, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 41, 9);
//        long timeToApplyAdaptiveThreshold = System.nanoTime() - timeToApplyAdaptiveThresholdStart;
//        Log.d(DEBUG_TAG, "Time to apply adaptive threshold (in ms): " + TimeUnit.MILLISECONDS.convert(timeToApplyAdaptiveThreshold, TimeUnit.NANOSECONDS));
//
//        long timeToWriteTransformedImageToStorageStart = System.nanoTime();
//        MatOfInt compressParams = new MatOfInt(  Imgcodecs.IMWRITE_PNG_BILEVEL, 1);
//        String fileNamePT = extStorageDirectory + "/" + fileName + "_" + "transformed.png";
//        imwrite(fileNamePT, thr, compressParams);
//        long timeToWriteTransformedImageToStorage = System.nanoTime() - timeToWriteTransformedImageToStorageStart;
//        Log.d(DEBUG_TAG, "Time to writing it to storage  (in ms): " + TimeUnit.MILLISECONDS.convert(timeToWriteTransformedImageToStorage, TimeUnit.NANOSECONDS));
        String fileNamePT = performAdaptiveThreshold(wrappedImage, fileName, 41, transformedFilePath);
        return fileNamePT + pointStr;
    }

    /**
     * Method to create thumbnail image for transformed image.
     *
     * @param imgFileNamePT
     * @return String thumbnail image path
     */
    public static String createThumbnailImage(String imgFileNamePT) {
        String thumbnailImagePath = "";
        try {

            Log.d(DEBUG_TAG, "createThumbnailImage calling.");
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            Bitmap imageBitmap = BitmapFactory.decodeFile(imgFileNamePT, options);

            final int THUMBNAIL_SIZE = 600;
            int width = imageBitmap.getWidth();
            int height = imageBitmap.getHeight();

            float bitmapRatio = (float) width / (float) height;
            if (bitmapRatio > 1) {
                width = THUMBNAIL_SIZE;
                height = (int) (width / bitmapRatio);
            } else {
                height = THUMBNAIL_SIZE;
                width = (int) (height * bitmapRatio);
            }
            imageBitmap = Bitmap.createScaledBitmap(imageBitmap, width, height, true);
            FileOutputStream outputStream;
            Log.d(DEBUG_TAG, "Bitmap image created.");
            try {
                String imageFileName = imgFileNamePT.substring(imgFileNamePT.indexOf("PT_IMG_"), imgFileNamePT.lastIndexOf(".png"));
                thumbnailImagePath = extStoragePublicDirectory + "/thumb_" + imageFileName + ".png";
                Log.d(DEBUG_TAG, "Thumbnail image path: " + thumbnailImagePath);
                OutputStream fOut = null;
                Integer counter = 0;
                File file = new File(thumbnailImagePath);

                fOut = new FileOutputStream(file);

                Bitmap pictureBitmap = imageBitmap; // obtaining the Bitmap
                pictureBitmap.compress(Bitmap.CompressFormat.PNG, 90, fOut); // saving the Bitmap to a file compressed as a JPEG with 85% compression rate
                fOut.flush();
                fOut.close();
                Log.d(DEBUG_TAG, "created ThumbnailImage saved.");
                //  MediaStore.Images.Media.insertImage(getContentResolver(),file.getAbsolutePath(),file.getName(),file.getName());

            } catch (Exception e) {
                e.printStackTrace();
                Log.d(DEBUG_TAG, e.getMessage());
                thumbnailImagePath = "";
            }
            return thumbnailImagePath;
        } catch (Exception ex) {
            ex.printStackTrace();
            Log.d(DEBUG_TAG, ex.getMessage());
            thumbnailImagePath = "";
        }
        return thumbnailImagePath;
    }

    /**
     * Method to perform adaptive threshold
     *
     * @param wrappedImage
     * @param fileName
     * @param thresholdBlocksize
     * @return
     */
    public static String performAdaptiveThreshold(Mat wrappedImage, String fileName, int thresholdBlocksize, String transformedFilePath) {

        Mat thresholdImg = new Mat();
//        Log.d(DEBUG_TAG, "Time to Converting image into gray (in ms): " + TimeUnit.MILLISECONDS.convert(timeToConvertImageToGray, TimeUnit.NANOSECONDS));
        long timeToApplyAdaptiveThresholdStart = System.nanoTime();
        Imgproc.adaptiveThreshold(wrappedImage, thresholdImg, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, thresholdBlocksize, 9);
        long timeToApplyAdaptiveThreshold = System.nanoTime() - timeToApplyAdaptiveThresholdStart;
        Log.d(DEBUG_TAG, "Time to apply adaptive threshold (in ms): " + TimeUnit.MILLISECONDS.convert(timeToApplyAdaptiveThreshold, TimeUnit.NANOSECONDS));

        long timeToWriteTransformedImageToStorageStart = System.nanoTime();
        MatOfInt compressParams = new MatOfInt(Imgcodecs.IMWRITE_PNG_BILEVEL, 1);
        //String appPath = "/data/data/oxs.eye/files"; //context.getFilesDir().getAbsolutePath();
        String fileNamePT = "";
        if (transformedFilePath == null || transformedFilePath.isEmpty()) {
            Random rand = new Random();
            // Generate random integers in range 0 to 999
            int randInt = rand.nextInt(100);
            fileNamePT = extStoragePublicDirectory + "/" + fileName + "_" + "transformed" + randInt + ".png";
        } else {
            fileNamePT = transformedFilePath + "/" + fileName + "_" + "transformed.png";
        }

        imwrite(fileNamePT, thresholdImg, compressParams);
        long timeToWriteTransformedImageToStorage = System.nanoTime() - timeToWriteTransformedImageToStorageStart;
        Log.d(DEBUG_TAG, "Time to writing it to storage  (in ms): " + TimeUnit.MILLISECONDS.convert(timeToWriteTransformedImageToStorage, TimeUnit.NANOSECONDS));

        return fileNamePT;
    }

    /**
     * Method to send email the source image and transformed image
     *
     * @param imgURI    The transformed image.
     * @param imgURISrc The source image
     * @return String the status of sent email
     */
    public static String sendEmail(String imgURI, String imgURISrc) {
        long timeToSendEmailStart = System.nanoTime();

        sendMailStatus = "The captured image sent by mail";
        attachmentFile = imgURI;
        attachmentFileSrc = imgURISrc;

        Uri URI = null;
        try {

            rec = "ccloud.dev18@gmail.com";
//            rec = "wmaas.dev@gmail.com";
            subject = "Test subject"; //sub.getText().toString();
            textMessage = "Test Message.."; //msg.getText().toString();

            Properties props = new Properties();
            props.put("mail.smtp.host", "smtp.gmail.com");
            props.put("mail.smtp.socketFactory.port", "465");
            props.put("mail.smtp.socketFactory.class", "javax.net.ssl.SSLSocketFactory");
            props.put("mail.smtp.auth", "true");
            props.put("mail.smtp.port", "465");

            session = Session.getDefaultInstance(props, new Authenticator() {
                protected PasswordAuthentication getPasswordAuthentication() {
//                    return new PasswordAuthentication("wmaas.dev@gmail.com", "Maasdev@789");
                    return new PasswordAuthentication("ccloud.dev18@gmail.com", "Chinna@123");
                }
            });

            RetreiveFeedTask task = new RetreiveFeedTask();
            task.execute();
            long timeToSendEmail = System.nanoTime() - timeToSendEmailStart;
            Log.d(DEBUG_TAG, "Time to initiate sending Email  (in ms): " + TimeUnit.MILLISECONDS.convert(timeToSendEmail, TimeUnit.NANOSECONDS));

        } catch (Exception e) {
            Log.d("SendMail", e.getMessage());
            sendMailStatus = "Request failed try again: " + e.toString();
            //       Toast.makeText(context, "Request failed try again: " + e.toString(), Toast.LENGTH_LONG).show();
        }
        return sendMailStatus;
        //  Mail mail = new Mail();
    }

    /**
     * Method to get file name from given image URI
     *
     * @param imgURI The source image URI location
     * @return String  the file name
     */
    public static String getFileName(String imgURI) {

        String prefixStr = "";

        // Assume the image URI starts with IMG_
        if (imgURI.indexOf("IMG_") > 0) {
            prefixStr = "IMG_";
        }

        // Check the file extension
        int fileExt = imgURI.indexOf(".jpeg");
        if (imgURI.indexOf(".jpg") > 0) {
            fileExt = imgURI.indexOf(".jpg");
        }

        // Sets the file name stars with "PT_"
        String fileName = "PT_";

        // Get the file name
        if (prefixStr == "") {
            fileName = fileName + imgURI.substring(imgURI.lastIndexOf('/') + 1, fileExt);
        } else {
            fileName = fileName + imgURI.substring(imgURI.indexOf(prefixStr), fileExt);
        }
        return fileName;
    }

    public void OpenCVLoaderInitAsync() {
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_1, context, mLoaderCallback);
    }

    /**
     * Draw lines on image
     * @param sortedRecpoints
     * @param imgURI
     * @param extStorageDirectory
     * @param fileName
     */
    private static void drawLinesOnImage(MatOfPoint2f sortedRecpoints, String imgURI, String extStorageDirectory, String fileName){
        Random rand = new Random();
        int r = rand.nextInt();
        int g = rand.nextInt();
        int b = rand.nextInt();
        Scalar color = new Scalar(r, g, b);
        Point[] points = sortedRecpoints.toArray();
        if(points.length > 3) {
            Mat srcMatOrg = imread(imgURI);

            line(srcMatOrg, new Point(points[0].x, points[0].y),
                    new Point(points[1].x, points[1].y),
                    color, 10);
            line(srcMatOrg, new Point(points[1].x, points[1].y),
                    new Point(points[2].x, points[2].y),
                    color, 10);
            line(srcMatOrg, new Point(points[2].x, points[2].y),
                    new Point(points[3].x, points[3].y),
                    color, 10);
            line(srcMatOrg, new Point(points[0].x, points[0].y),
                    new Point(points[3].x, points[3].y),
                    color, 10);
            String fileNameContour = extStorageDirectory + "/" + fileName + "_contour.jpg";

            imwrite(fileNameContour, srcMatOrg);
            srcMatOrg = null;
        }
    }

    /**
     * Private class to send email asynchronously
     */
    private static class RetreiveFeedTask extends AsyncTask<String, Void, String> {

        @Override
        protected String doInBackground(String... params) {
            //  String logFileName = params[0];

            try {
                long timeToExecuteSendEmailStart = System.nanoTime();
                // String filename = attachmentFile;
                int idxEnd = attachmentFile.indexOf(".png");
                if (idxEnd < 0)
                    idxEnd = attachmentFile.indexOf(".jpg");

                String fileName = attachmentFile.substring(attachmentFile.indexOf("PT_"), idxEnd);

                Message message = new MimeMessage(session);
//                message.setFrom(new InternetAddress("wmaas.dev@gmail.com"));
                message.setFrom(new InternetAddress("ccloud.dev18@gmail.com"));
                message.setRecipients(Message.RecipientType.TO, InternetAddress.parse(rec));
                message.setSubject(fileName + ".png");
                message.setContent(textMessage, "text/html; charset=utf-8");

                // Create the message part
                BodyPart messageBodyPart = new MimeBodyPart();

                //messageBodyPart.setDataHandler(new DataHandler());
                // Now set the actual message
                messageBodyPart.setText("Please check the above file as attachment. Thanks");

                // Create a multipar message
                Multipart multipart = new MimeMultipart();

                // Set text message part
                multipart.addBodyPart(messageBodyPart);

                // Part two is attachment
                messageBodyPart = new MimeBodyPart();

                DataSource source = new FileDataSource(attachmentFile);
                messageBodyPart.setDataHandler(new DataHandler(source));
                messageBodyPart.setFileName(attachmentFile);
                multipart.addBodyPart(messageBodyPart);

                //Adding source image
                messageBodyPart = new MimeBodyPart();

                source = new FileDataSource(attachmentFileSrc);
                messageBodyPart.setDataHandler(new DataHandler(source));
                messageBodyPart.setFileName(attachmentFileSrc);
                multipart.addBodyPart(messageBodyPart);

                //Adding Log file
                messageBodyPart = new MimeBodyPart();

                source = new FileDataSource(logFile);
                messageBodyPart.setDataHandler(new DataHandler(source));
                messageBodyPart.setFileName(logFile);
                multipart.addBodyPart(messageBodyPart);


                // Send the complete message parts
                message.setContent(multipart);

                Transport.send(message);
                long timeToExecuteSendEmail = System.nanoTime() - timeToExecuteSendEmailStart;
                Log.d(DEBUG_TAG, "Time to execute sending Email  (in ms) : " + TimeUnit.MILLISECONDS.convert(timeToExecuteSendEmail, TimeUnit.NANOSECONDS));
                String logFileToBedeleted = getFileName(attachmentFileSrc);
                deleteLogs(logFileToBedeleted);

            } catch (MessagingException e) {
                sendMailStatus = e.getMessage();
                e.printStackTrace();
            } catch (Exception e) {
                sendMailStatus = e.getMessage();
                e.printStackTrace();
            }
            return sendMailStatus;
        }

        @Override
        protected void onPostExecute(String result) {
            sendMailStatus = result;
        }
    }
}

