#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '..')
cwd = os.getcwd() + "/src/Grounded-SAM-2"
sys.path.insert(0, cwd)
import rclpy
import torch
import re
import yaml
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from rclpy.node import Node
from std_msgs.msg import String, Int64
from dataclasses import dataclass
import supervision as sv
from cv_bridge import CvBridge
import numpy as np
from scipy.spatial.transform import Rotation
from typing import List
from sklearn.cluster import KMeans
from grounding_dino.groundingdino.util.inference import Model
from PIL import Image
from athena_msgs.srv import DetectObjects
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose



GROUNDING_DINO_CONFIG = os.path.join(cwd, "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT = os.path.join(cwd, "gdino_checkpoints/groundingdino_swint_ogc.pth")
SAM2_CHECKPOINT = os.path.join(cwd,"checkpoints/sam2.1_hiera_large.pt")
SAM2_MODEL_CONFIG =  "configs/sam2.1/sam2.1_hiera_l.yaml"
OUT_LOCATION = [0.2, 0.65, 0.5, -0.610608, 0.791903, 0.0, 0.0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def exists(var):
    return not (var is None)



def extract_objects_from_pddl_text(pddl_text):
    init_start = pddl_text.index(':init')
    goal_start = pddl_text.index(':goal')

    init_obj_wps = re.findall(r"\(\S+ (\S+) (\S+)\)", pddl_text[init_start:goal_start])
    goal_obj_wps = re.findall(r"\(\S+ (\S+) (\S+)\)", pddl_text[goal_start:])

    object_dictionary = {}
    for objs in init_obj_wps:
        object_dictionary[objs[0]] = {'wps': objs[1]}

    for objs in goal_obj_wps:
        if objs[0] in object_dictionary:
            object_dictionary[objs[0]]['wpf'] = objs[1]

    
    new_object_dictionary = {key: value for key, value in object_dictionary.items() if "robot" not in key}
    list_of_objects = ["".join([ch for ch in obj_name if not ch.isnumeric()]) for obj_name in new_object_dictionary.keys()]
    unq_objs, counts = np.unique(list_of_objects, return_counts=True)
    unq_objs = unq_objs.tolist()
    return new_object_dictionary, unq_objs, counts

@dataclass
class ModelConfig:
    GROUNDING_DINO_CONFIG: str 
    GROUNDING_DINO_CHECKPOINT: str 


class GibsonSAM:
    BOX_TRESHOLD = 0.40
    TEXT_TRESHOLD = 0.25
    MIN_IMAGE_AREA_PERCENTAGE = 0.002
    MAX_IMAGE_AREA_PERCENTAGE = 0.80
    APPROXIMATION_PERCENTAGE = 0.75
    def __init__(self, model_config, device: str = "cuda"):
        self.device = device
        self.model_config = model_config
        self.model_config = model_config
        self.box_annotator = sv.BoxAnnotator()
        self.mask_annotator = sv.MaskAnnotator()
        self.grounding_dino_model = Model(model_config_path=self.model_config.GROUNDING_DINO_CONFIG, model_checkpoint_path=self.model_config.GROUNDING_DINO_CHECKPOINT)

        self.sam2_checkpoint = SAM2_CHECKPOINT
        self.model_cfg = SAM2_MODEL_CONFIG
        self.sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device=DEVICE)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)


    @staticmethod
    def segment(sam2_predictor: SAM2ImagePredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam2_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False,
)
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)
    

    @staticmethod
    def enhance_class_name(class_names: List[str]) -> List[str]:
        return [
            f"{class_name}"
            for class_name
            in class_names
    ]

    
    def get_segmentation(self, image, class_names): 
        detections = self.grounding_dino_model.predict_with_classes(
                image=image,
                classes=self.enhance_class_name(class_names),
                box_threshold=self.BOX_TRESHOLD,
                text_threshold=self.TEXT_TRESHOLD
            )
        detections = detections[detections.class_id != None]
        
        detections.mask = self.segment(
            self.sam2_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy )
        return detections.mask,  [class_names[class_id]for class_id in detections.class_id]
        

config = ModelConfig(GROUNDING_DINO_CONFIG=GROUNDING_DINO_CONFIG,
                     GROUNDING_DINO_CHECKPOINT=GROUNDING_DINO_CHECKPOINT
                    )

class ObjectDetector:
    def __init__(self):
        self.segmentation = GibsonSAM(model_config= config)

        # self.panda_T_camera = panda_T_marker @ np.linalg.inv(camera_T_marker)
        self.panda_T_camera = np.array([[-0.00183383,  0.75774677, -0.65252969,  1.10053929],
                                        [ 0.99976673,  0.01625902,  0.0151943 ,  0.20608223],
                                        [ 0.02153863, -0.65234401, -0.75761344,  0.93254739],
                                        [ 0.        ,  0.        ,  0.        ,  1.        ]])
       
        self.width = 640
        self.height = 480
        self.cx= 320.5542297363281 
        self.cy= 239.13539123535156
        self.fx= 390.49041748046875 
        self.fy = 390.49041748046875   
        self.depth_scale = 0.0010000000474974513

    @staticmethod
    def get_surface_normal_direction(pc):
        ## only in 2D, robot xy plane is parallel to table
        mean = pc[:,:2].mean(axis=0)
        minidx = np.linalg.norm(pc[:,:2]-mean).argmin()
        direction = pc[minidx,:2] - mean
        return direction/np.linalg.norm(direction)
    
    @staticmethod
    def get_grasp_pose(pc): # pc in robot frame 
        dim = pc.max(axis=0)-pc.min(axis=0)
        delta = dim.min()
        
        # Get Coordinate Axis for the object
        median = np.median(pc, axis=0)
        mean = np.mean(pc, axis=0)
        diff = pc[:,:2] - mean[:2] #along x and y axis 
        cov = np.cov(diff.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        sort_idx = np.argsort(eigvals)

        # y-direction to be corrected using surface normal direction
        direction = ObjectDetector.get_surface_normal_direction(pc)
        cos_t = np.dot(direction, eigvecs[:,sort_idx[0]])
        if cos_t<0:
            eigvecs[:,sort_idx[0]] *= -1
        
        rot = np.eye(3)*-1 #grazp z is downwards, robot z is upwards
        rot[:2,:2] = eigvecs[:, [sort_idx[1], sort_idx[0]]] #grasp y is hand closing direction should be along small chnage
        
        mean_rot = rot.T@mean + np.array([0,delta*0.5,0])
        median_rot = rot.T@median + np.array([0,delta*0.5,0])

        # return rotmean, median, rot
        return rot@mean_rot, rot@median_rot, rot
    
    def performObjectDetection(self, color_image, depth_image, object_names):
        depth_image = depth_image.astype(float)*self.depth_scale
        masks, classes = self.segmentation.get_segmentation(color_image, object_names)

        objects_dict = {}
        for mask, cls in zip(masks, classes):
            indexes = np.argwhere(mask)

            z = depth_image[indexes[:, 0], indexes[:, 1]]
            x = (indexes[:,1].astype(float) - self.cx)*z/self.fx
            y = (indexes[:,0].astype(float) - self.cy)*z/self.fy

            pc = np.hstack((x[:,None], y[:,None], z[:,None]))[z>0.15]
            pc_color = color_image[indexes[:,0], indexes[:,1], ::-1][z>0.15]


            kmean = KMeans(n_clusters=2).fit(pc)
            unq, counts = np.unique(kmean.labels_)
            pc = pc[kmean.labels_==counts.argmax()]
            pc_color = pc_color[kmean.labels_==counts.argmax()]

            ### Transform to Panda
            pc_panda = np.einsum("ij, kj->ki", self.panda_T_camera[:3,:3], pc) + self.panda_T_camera[:3,3]

            # Compute the grasp pose estimation
            mean_org, median_org, rot_org = self.get_grasp_pose(pc_panda.copy())

            panda_hand_position = mean_org-np.array([0,0.0,-0.058])
            if panda_hand_position[2]<0.43:
                panda_hand_position[2] = 0.43

            objects_dict[cls] = {'translation': panda_hand_position.tolist(), 'quaternion': Rotation.from_matrix(rot_org).as_quat().tolist()}

        for name in object_names:
            if not (name in objects_dict):
                objects_dict[name] = {'translation': [0.0,0.0,0.0], 'quaternion': [0.0,0.0,0.0,1.0]}
                #print(f"\033[31m{name} not detected!\033[0m")

        return objects_dict
        


class ObjectDetectorNode (Node):
    def __init__(self):


        super().__init__('pose_estimator_service_service_node')
        self.pose_estimator = ObjectDetector()
        self.get_logger().info('Model Loaded!')
        self.color_image_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_camera_callback, 10)
        self.depth_image_sub = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_camera_callback, 10)
        self.srv = self.create_service(DetectObjects, 'detect_objects', self.pose_estimator_callback)
        self.color_image = None
        self.depth_image = None
        self.object_dictionary = None
        self.unq_objs = None
        self.obj_counts = None

    def color_camera_callback(self, msg):
        bridge = CvBridge()
        self.color_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        #cv2.imshow("Camera Image", self.color_image)
        #cv2.waitKey(1) 

    def depth_camera_callback(self, msg):
        bridge = CvBridge()
        self.depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        #cv2.imshow("Depth Image", depth_image_normalized)
        #cv2.waitKey(1)

   
                

    def pose_estimator_callback(self, request, response):

        #Get the list of objects from the PDDL file
        pddl_text = request.problem_file.data
        self.get_logger().info(pddl_text)
        self.object_dictionary, self.unq_objs, self.obj_counts = extract_objects_from_pddl_text(pddl_text)
        

        self.get_logger().info(f'Extracted objects: {list(self.object_dictionary)}')
        objects_names = []
        objects_positions = []
        #return response
        if exists(self.color_image) and exists(self.depth_image):
            object_names = [obj for obj in self.object_dictionary.keys()]
            objects_dict = self.pose_estimator.performObjectDetection(self.color_image, self.depth_image, object_names)
            for k, v in objects_dict.items():
                #Get object name
                obj = String()
                obj.data = k
                #Get count
                count = Int64()
                
                #Get object position
                pose = Pose()
                x, y, z = v['translation']
                qx, qy, qz, qw = v['quaternion']
                pose.position.x = x
                pose.position.y = y
                pose.position.z = z
                pose.orientation.x = qx
                pose.orientation.y = qy
                pose.orientation.z = qz
                pose.orientation.w = qw
    
                objects_names.append(k)
                objects_positions.append(pose)

                response.objects_names.append(obj)
                response.objects_counts.append(count)
                response.objects_positions.append(pose)
                
                if (x > 0 and y>=0 and z>0):
                    self.get_logger().info('Object %s Pose:(%0.2f,%0.2f,%0.2f), Orientation:(%0.2f,%0.2f,%0.2f, %0.2f) ' 
                    % (k, x , y, z, qx , qy, qz,qw ))
                else:
                    self.get_logger().error('Object %s not found!' %k)


            self.save_locations_to_file(objects_names, objects_positions)
            return response
        else:
            self.get_logger().error('Camera Image not available!')
            return response
        

    def save_locations_to_file(self, objects_names, objects_positions):

        ### Check and save the object counts from LLM and GroundingSAM 
        objects_detected_names = [obj_name for obj_name in objects_names]
     
        unq_objects_detected, objects_detected_counts = np.unique(objects_detected_names, return_counts=True)
        unq_objects_detected = unq_objects_detected.tolist()
  
        object_tally_dict = {}
        for obj_name_llm, obj_count_llm in zip(self.unq_objs, self.obj_counts):
            if obj_name_llm in unq_objects_detected:
                object_tally_dict[obj_name_llm] = {"LLM": int(obj_count_llm),
                                                   "SAM": int(objects_detected_counts[unq_objects_detected.index(obj_name_llm)])}
            else:
                object_tally_dict[obj_name_llm] = {"LLM": int(obj_count_llm),
                                                   "SAM": 0}
        
        with open("object_tally.yaml", "w") as f:
            yaml.dump(object_tally_dict, f, default_flow_style=False)
        
        
        ### Mapping between objects and their location using dictionary
        objects_poses = {}
        objects_added = []
        for obj_name, obj_pose in zip(objects_names, objects_positions):
           
            count = objects_detected_counts[unq_objects_detected.index(obj_name)]
            if count==1:
                add = ""
            else:
                objects_added.append(obj_name)
                add = str(objects_added.count(obj_name))


            objects_poses[obj_name+add] = [obj_pose.position.x,
                                                obj_pose.position.y,
                                                obj_pose.position.z,
                                                obj_pose.orientation.x,
                                                obj_pose.orientation.y,
                                                obj_pose.orientation.z,
                                                obj_pose.orientation.w]


        ### Assigning positions wp1s...., wp1f...., object
        object_dictionary_reversed = {}
        for name, way_pnts in self.object_dictionary.items():
            if name in objects_poses:
                wps = way_pnts['wps']
                wpf = way_pnts['wpf']
                if not (re.search('wp\wf', wpf) is None): ### final position is same as initial position wp1f or wp2f etc..
                    object_dictionary_reversed[name] = {wps: objects_poses[name],
                                                        wpf: objects_poses[name]}
                    
                elif wpf in objects_poses: ### final position is top of object
                    object_dictionary_reversed[name] = {wps: objects_poses[name],
                                                        wpf: objects_poses[wpf]}
                    
                elif 'out' in wpf: ### or its an out location
                    object_dictionary_reversed[name] = {wps: objects_poses[name],
                                                        wpf: self.out_location}
        
        writing_everthing_in_yaml = []
        done = []
        for name, way_pnts in object_dictionary_reversed.items():
            if not (name in done):
                for k, v in self.object_dictionary[name].items():
                    if (k=='wps') and (not (v in done)):
                        pose_str = ','.join([f'{p:0.4f}' for p in way_pnts[v]])
                        writing_everthing_in_yaml.append(name + ': ' + '[' + pose_str + ']')
                        writing_everthing_in_yaml.append(v + ': ' + '[' + pose_str + ']')
                        done.extend([name, v])
                        
                    if (k=='wpf') and (not (v in done)):
                        if not (re.search('wp\wf', v) is None): ### final position is same as initial position wp1f or wp2f etc..
                            pose_str = ','.join([f'{p:0.4f}' for p in way_pnts[v]])
                            writing_everthing_in_yaml.append(v + ': ' + '[' + pose_str + ']')
                            done.append(v)
                            
                        elif v in objects_poses: ### final position is top of object
                            pose_str = ','.join([f'{p:0.4f}' for p in objects_poses[v]])
                            writing_everthing_in_yaml.append(v + ': ' + '[' + pose_str + ']')
                            done.append(v)
                            
                        elif 'out' in v: ### or its an out location
                            pose_str = ','.join([f'{p:0.4f}' for p in self.out_location])
                            writing_everthing_in_yaml.append(v + ': ' + '[' + pose_str + ']')
                            done.append(v)
                        
        with open('way_points.yaml', 'w') as f:
            f.write('\n'.join(writing_everthing_in_yaml))

def main(args=None):

    #pose_estimator = ObjectDetector()
    #pose_estimator.run()
    #humnadet.collect()

    rclpy.init(args=args)

    pose_estimator_service = ObjectDetectorNode()

    rclpy.spin(pose_estimator_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
