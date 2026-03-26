string label               # "X" or "O"
float32 confidence

# 2D bbox in pixels (image coordinates)
int32 xmin
int32 ymin
int32 xmax
int32 ymax

# 2D center pixel (optional but useful)
int32 cx
int32 cy

# 3D center in some frame (camera/base/board)
geometry_msgs/PointStamped center_3d

# Optional orientation info (keep simple for now)
float32 yaw_deg
