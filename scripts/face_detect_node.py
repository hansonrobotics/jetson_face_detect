#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from face_detect.msg import FaceDetectionResult
import jetson_inference
import jetson_utils
import cv2
from cv_bridge import CvBridge, CvBridgeError
from ddynamic_reconfigure_python.ddynamic_reconfigure import DDynamicReconfigure


class FaceDetectionNode:
    def __init__(self):
        self.bridge = CvBridge()
        
        self.net = jetson_inference.detectNet("facedetect", threshold=0.5)
        self.net.SetTrackingEnabled(True)
        self.net.SetTrackerType('IOU')
        self.net.SetClusteringThreshold(0.5)

        print(f"confidence_threshold: {self.net.GetConfidenceThreshold()}")
        print(f"clustering_threshold: {self.net.GetClusteringThreshold()}")
        print(f"tracking: {self.net.IsTrackingEnabled()}")
        print(f"tracking_params: {self.net.GetTrackingParams()}")
        self.cfg = None
        ddynrec = DDynamicReconfigure("eye_camera")
        ddynrec.add_variable("eye_center_x", "horizontal Eye center ", float(0.0), -0.99, 0.99)
        ddynrec.add_variable("eye_center_y", "Vertical Eye center ", float(0.0), -0.99, 0.99)
        ddynrec.start(self.update_cfg)
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)
        self.face_detect_pub = rospy.Publisher("face_detect/detect_results", FaceDetectionResult, queue_size=30)
        ddynrec = DDynamicReconfigure("eye_camera")

        self.eye_image_debug = rospy.Publisher("eye_camera_img", Image, queue_size=30)

    def update_cfg(self, config, level=None):
        self.cfg = config
        return config

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "rgba8")
            cuda_image = jetson_utils.cudaFromNumpy(cv_image)
            
            '''
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            bgr_img = jetson_utils.cudaFromNumpy(cv_image)
            cuda_image = jetson_utils.cudaAllocMapped(width=bgr_img.width, height=bgr_img.height, format='rgb8')
            jetson_utils.cudaConvertColor(bgr_img, cuda_image)
            '''

            detections = self.net.Detect(cuda_image)

            for detection in detections:
                face_detection_result = FaceDetectionResult()
                face_detection_result.timestamp = data.header.stamp
                face_detection_result.track_id = detection.TrackID
                face_detection_result.left = detection.Left
                face_detection_result.top = detection.Top
                face_detection_result.right = detection.Right
                face_detection_result.bottom = detection.Bottom
                self.face_detect_pub.publish(face_detection_result)
            
            cv_image = jetson_utils.cudaToNumpy(cuda_image)
            cv2.circle(cv_image, (int(320+320*self.cfg.eye_center_x), int(240+240*self.cfg.eye_center_y)), 4, (255,0,0), 1)
            cv2.imshow("USB Camera Stream with Detections", cv_image)
            cv2.waitKey(1)
                
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

def main():
    rospy.init_node('face_detect_node')
    fd_node = FaceDetectionNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()