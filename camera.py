import depthai as dai
import numpy as np

# Initialize pipeline
pipeline = dai.Pipeline()

# Configure left and right mono cameras
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# Create stereo depth node
stereo = pipeline.createStereoDepth()
stereo.initialConfig.setConfidenceThreshold(200)
left.out.link(stereo.left)
right.out.link(stereo.right)

# Create XLinkOut node to output point cloud
xout_pc = pipeline.createXLinkOut()
xout_pc.setStreamName("point_cloud")
stereo.disparity.link(xout_pc.input)

# Start the pipeline
with dai.Device(pipeline) as device:
    # Get the point cloud output queue
    point_cloud_queue = device.getOutputQueue(name="point_cloud", maxSize=4, blocking=False)

    while True:
        # Get the point cloud data
        point_cloud_data = point_cloud_queue.get()
        if point_cloud_data is not None:
            # Process point cloud data
            points = np.array(point_cloud_data.getData()).view(np.float32).reshape((480, 640, 3))
            # Now you can use 'points' to do further processing or visualization
