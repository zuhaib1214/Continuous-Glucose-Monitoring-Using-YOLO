import cv2
import asyncio
import websockets
import json
import base64
import numpy as np
import time

# Configuration
SERVER_URL = "ws://localhost:8000/ws"
CAM_INDEX = 0
TARGET_FPS = 30
frame_interval = 1.0 / TARGET_FPS


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


async def send_frames(websocket):
    """Send frames from camera to WebSocket"""
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    
    last_time = time.time()
    frame_count = 0
    
    print("Starting camera capture...")
    print("Press 'q' to quit.")
    
    try:
        while True:
            now = time.time()
            dt = now - last_time
            
            # Frame rate limiting
            if dt < frame_interval:
                await asyncio.sleep(frame_interval - dt)
            
            last_time = time.time()
            
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break
            
            # Encode frame
            frame_base64 = encode_frame(frame)
            
            # Send frame to server
            try:
                await websocket.send(frame_base64)
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Sent {frame_count} frames")
            except Exception as e:
                print(f"Error sending frame: {e}")
                break
                
    except Exception as e:
        print(f"Error in send_frames: {e}")
    finally:
        cap.release()
        print("Camera released")


async def receive_frames(websocket):
    """Receive processed frames from WebSocket"""
    try:
        while True:
            # Receive response from server
            response = await websocket.recv()
            
            try:
                # Parse JSON response
                data = json.loads(response)
                
                if "error" in data:
                    print(f"Server error: {data['error']}")
                    continue
                
                # Decode processed frame
                processed_frame_base64 = data.get("frame", "")
                if processed_frame_base64:
                    processed_frame = decode_frame(processed_frame_base64)
                    
                    if processed_frame is not None:
                        # Display frame
                        cv2.imshow("Processed Frame (press 'q' to quit)", processed_frame)
                        
                        # Print outputs if available
                        outputs = data.get("outputs", [])
                        if outputs:
                            print(f"Detections: {outputs}")
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quit key pressed")
                    break
                    
            except json.JSONDecodeError:
                print("Failed to parse JSON response")
                continue
            except Exception as e:
                print(f"Error processing response: {e}")
                continue
                
    except Exception as e:
        print(f"Error in receive_frames: {e}")
    finally:
        cv2.destroyAllWindows()


async def main():
    """Main function to connect and handle WebSocket communication"""
    print(f"Connecting to {SERVER_URL}...")
    
    try:
        async with websockets.connect(SERVER_URL) as websocket:
            print("Connected to WebSocket server!")
            
            # Create tasks for sending and receiving
            send_task = asyncio.create_task(send_frames(websocket))
            receive_task = asyncio.create_task(receive_frames(websocket))
            
            # Wait for either task to complete
            done, pending = await asyncio.wait(
                [send_task, receive_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed by server")
    except Exception as e:
        print(f"Connection error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cv2.destroyAllWindows()

