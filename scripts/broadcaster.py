#!/usr/bin/env python
import numpy as np
import tf.transformations
import pickle
import rospy
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Transform
import multiprocessing

def load_matrix(filename):
    f = open("transformation_data/" + filename, "rb")
    matrix = np.matrix(pickle.load(f))
    f.close()
    return matrix

def get_transform_from_file(filename):
    matrix = load_matrix(filename)
    rotation = np.zeros((4, 4))
    rotation[3, 3] = 1
    rotation[:3, :3] = matrix[:3,:3]
    translation = matrix[:,3].T.tolist()[0]
    quaternion = tf.transformations.quaternion_from_matrix(rotation)
    quaternion = Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
    pose = Vector3(translation[0], translation[1], translation[2])
    return Transform(pose, quaternion)

def get_transform_from_matrix(matrix):
    """
    Takes in a 3x4 rotation + translation matrix and outputs a ROS Transform object.
    """
    rotation = np.zeros((4, 4))
    rotation[3, 3] = 1
    rotation[:3, :3] = matrix[:3,:3]
    translation = matrix[:,3].T.tolist()[0]
    quaternion = tf.transformations.quaternion_from_matrix(rotation)
    quaternion = Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
    pose = Vector3(translation[0], translation[1], translation[2])
    return Transform(pose, quaternion)

def transform_publisher(name, transform):
    """
    Creates a transform publisher that publishes the Ros Transform transform with the node called transforms/name.
    """
    pub = rospy.Publisher('transforms/' + name, Transform, queue_size=10)
    rospy.init_node('transform_publisher', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        pub.publish(transform)
        rate.sleep()

def matrix_from_transform(transform):
    """
    Takes in a ROS Transform, outputs a 3x4 rotation + translation matrix.
    """
    mat = np.zeros((3,4))
    mat[:3,:3] = tf.transformations.quaternion_matrix([transform.rotation.x,transform.rotation.y,transform.rotation.z,transform.rotation.w])[:3,:3]
    mat[:,3] = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
    return mat

if __name__ == '__main__':
    try:
        names = ["psm1_to_psm2", "psm2_to_psm1", "psm1_to_endoscope", "psm2_to_endoscope", "endoscope_to_psm1", "endoscope_to_psm2"]
        matrices = [np.matrix([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0]]),
            np.matrix([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0]]),
            np.matrix([[-0.98313063, -0.13662154,  0.12160893, -0.08522398], [-0.1335421,   0.99048819,  0.03316115, -0.05509822], [-0.12498274,  0.01636183, -0.992024, 0.00896234]]),
            np.matrix([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0]]),
            np.matrix([[-0.98313062, -0.1335421,  -0.12498273,  0.08522398], [-0.13662154,  0.99048818,  0.01636183,  0.05509822], [0.12160893,  0.03316115, -0.99202399, -0.00896234]]),
            np.matrix([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0]])]
            

        print matrices[2]
        print matrices[4]

        transforms = [get_transform_from_matrix(matrix) for matrix in matrices]

        for i in range(len(transforms)):
            prs = multiprocessing.Process(target=transform_publisher, args=[names[i], transforms[i]])
            prs.start()
        print "Successfully started all nodes."

    except rospy.ROSInterruptException:
        pass
