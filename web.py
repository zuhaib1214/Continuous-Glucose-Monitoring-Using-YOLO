from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import torch
from ultralytics import YOLO
import asyncio
from typing import Optional

# Configuration
MODEL_PATH = "yolov8n-pose.pt"
CONF_THR = 0.50
KP_THR = 0.70
IMGSZ = 320
RIGHT_SHOULDER, LEFT_SHOULDER = 5, 6
RIGHT_ELBOW, LEFT_ELBOW = 7, 8

# Initialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)
model.to(device)

# Drawing colors
line_color = (0, 255, 255)
box_color = (255, 128, 0)
shoulder_color = (0, 255, 0)
elbow_color = (0, 0, 255)
cgm_color = (147, 20, 255)
text_color = (255, 255, 0)
font = cv2.FONT_HERSHEY_SIMPLEX

# FastAPI app
app = FastAPI(title="CGM Pose Detection WebSocket Server")


def get_33_point(p1, p2):
    """Calculate CGM point at 33% from shoulder to elbow"""
    x = p1[0] + 0.33 * (p2[0] - p1[0])
    y = p1[1] + 0.33 * (p2[1] - p1[1])
    return int(x), int(y)


def seg_angle_deg(p1, p2):
    """Calculate angle in degrees between two points"""
    dx = (p2[0] - p1[0])
    dy = (p2[1] - p1[1])
    return float(np.degrees(np.arctan2(dy, dx)))


def arm_output(kp_xy, kp_conf, shoulder_idx, elbow_idx):
    """Process arm keypoints and return output data"""
    if kp_xy is None:
        return False, None, None
    s = kp_xy[shoulder_idx]
    e = kp_xy[elbow_idx]
    if kp_conf is not None:
        s_conf = float(kp_conf[shoulder_idx])
        e_conf = float(kp_conf[elbow_idx])
        conf = min(s_conf, e_conf)
    else:
        conf = 1.0
    if np.any(np.isnan(s)) or np.any(np.isnan(e)):
        return False, None, None
    if conf < KP_THR:
        return False, None, None
    s_i = (int(s[0]), int(s[1]))
    e_i = (int(e[0]), int(e[1]))
    cgm = get_33_point(s_i, e_i)
    ang = seg_angle_deg(s_i, e_i)
    out = [int(cgm[0]), int(cgm[1]), round(conf, 3), round(ang, 2)]
    draw = {"shoulder": s_i, "elbow": e_i, "cgm": cgm}
    return True, out, draw


def process_frame(frame: np.ndarray) -> tuple[np.ndarray, list]:
    """
    Process a single frame: detect poses, draw annotations, return processed frame and outputs
    """
    # Run YOLO prediction
    results = model.predict(
        source=frame,
        conf=CONF_THR,
        imgsz=IMGSZ,
        verbose=False,
        device=device
    )
    
    per_frame_outputs = []
    
    # Process each detected person
    for r in results:
        if r.keypoints is None:
            continue
        kp_xy_all = r.keypoints.xy
        kp_cf_all = getattr(r.keypoints, "conf", None)
        
        for i in range(len(kp_xy_all)):
            kp_xy = kp_xy_all[i].cpu().numpy()
            kp_cf = None
            if kp_cf_all is not None:
                kp_cf = kp_cf_all[i].cpu().numpy()
            
            outputs_person = {}
            
            # Process right arm
            ok_r, out_r, draw_r = arm_output(kp_xy, kp_cf, RIGHT_SHOULDER, RIGHT_ELBOW)
            if ok_r:
                cv2.line(frame, draw_r["shoulder"], draw_r["elbow"], line_color, 2)
                cv2.circle(frame, draw_r["shoulder"], 3, shoulder_color, -1)
                cv2.circle(frame, draw_r["elbow"], 3, elbow_color, -1)
                cv2.circle(frame, draw_r["cgm"], 5, cgm_color, -1)
                x_min = min(draw_r["shoulder"][0], draw_r["elbow"][0]) - 5
                y_min = min(draw_r["shoulder"][1], draw_r["elbow"][1]) - 5
                x_max = max(draw_r["shoulder"][0], draw_r["elbow"][0]) + 5
                y_max = max(draw_r["shoulder"][1], draw_r["elbow"][1]) + 5
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                pos = (draw_r["cgm"][0] + 15, draw_r["cgm"][1] - 10)
                cv2.putText(frame, "Right Arm", pos, font, 0.5, text_color, 1, cv2.LINE_AA)
                outputs_person["right_arm"] = out_r
            
            # Process left arm
            ok_l, out_l, draw_l = arm_output(kp_xy, kp_cf, LEFT_SHOULDER, LEFT_ELBOW)
            if ok_l:
                cv2.line(frame, draw_l["shoulder"], draw_l["elbow"], line_color, 2)
                cv2.circle(frame, draw_l["shoulder"], 3, shoulder_color, -1)
                cv2.circle(frame, draw_l["elbow"], 3, elbow_color, -1)
                cv2.circle(frame, draw_l["cgm"], 5, cgm_color, -1)
                x_min = min(draw_l["shoulder"][0], draw_l["elbow"][0]) - 5
                y_min = min(draw_l["shoulder"][1], draw_l["elbow"][1]) - 5
                x_max = max(draw_l["shoulder"][0], draw_l["elbow"][0]) + 5
                y_max = max(draw_l["shoulder"][1], draw_l["elbow"][1]) + 5
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                pos = (draw_l["cgm"][0] - 120, draw_l["cgm"][1] - 10)
                cv2.putText(frame, "Left Arms", pos, font, 0.5, text_color, 1, cv2.LINE_AA)
                outputs_person["left_arm"] = out_l
            
            if outputs_person:
                per_frame_outputs.append(outputs_person)
    
    # Add FPS text
    cv2.putText(frame, "CGM Pose Detection", (10, 25), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    return frame, per_frame_outputs


def encode_frame(frame: np.ndarray) -> str:
    """Encode frame to base64 string"""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return frame_base64


def decode_frame(frame_data: str) -> np.ndarray:
    """Decode base64 string to frame"""
    frame_bytes = base64.b64decode(frame_data)
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    return frame


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "CGM Pose Detection WebSocket Server"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "device": device,
            "model_loaded": True
        }
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for frame streaming"""
    await websocket.accept()
    print("Client connected")
    
    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()
            
            try:
                # Decode frame from base64
                frame = decode_frame(data)
                
                if frame is None:
                    await websocket.send_json({"error": "Failed to decode frame"})
                    continue
                
                # Process frame
                processed_frame, outputs = process_frame(frame)
                
                # Encode processed frame
                processed_frame_base64 = encode_frame(processed_frame)
                
                # Send back processed frame and outputs
                response = {
                    "frame": processed_frame_base64,
                    "outputs": outputs
                }
                await websocket.send_json(response)
                
            except Exception as e:
                error_msg = f"Processing error: {str(e)}"
                print(error_msg)
                await websocket.send_json({"error": error_msg})
                continue
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)

