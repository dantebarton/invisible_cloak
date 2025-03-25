#!/usr/bin/env python3
"""
Invisible Cloak

This script implements a "Harry Potter-like" invisible cloak effect using 
computer vision techniques. It captures a background frame, then replaces
pixels matching a specific color range with the background, creating an
illusion of invisibility.
"""

import cv2
import time
import numpy as np
import sys
from contextlib import contextmanager

# Configuration variables
CAMERA_INDEX = 0
WINDOW_NAME = "Invisible Cloak"
BACKGROUND_CAPTURE_DELAY = 2  # seconds
EXIT_KEY = 'q'

# Color range for the "invisible" material (green by default)
# HSV format: Hue, Saturation, Value
LOWER_COLOR = np.array([50, 80, 50])     # Lower bound HSV for green
UPPER_COLOR = np.array([90, 255, 255])   # Upper bound HSV for green

# Morphological operation parameters
MORPH_KERNEL_SIZE = (5, 5)
MORPH_KERNEL = np.ones(MORPH_KERNEL_SIZE, np.uint8)


@contextmanager
def camera_capture(camera_index=0):
    """
    Context manager for handling camera capture resources.
    
    Args:
        camera_index: Index of the camera to use
        
    Yields:
        cv2.VideoCapture object if successful
        
    Raises:
        RuntimeError: If camera cannot be opened
    """
    cap = cv2.VideoCapture(camera_index)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera at index {camera_index}")
        yield cap
    finally:
        cap.release()
        cv2.destroyAllWindows()


def filter_mask(mask):
    """
    Apply morphological operations to improve mask quality.
    
    Args:
        mask: Binary mask to be filtered
        
    Returns:
        Filtered mask with noise removed
    """
    # CLOSE operation (dilation followed by erosion) removes black noise
    close_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
    
    # OPEN operation (erosion followed by dilation) removes white noise
    open_mask = cv2.morphologyEx(close_mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    
    return open_mask


def capture_background(cap):
    """
    Capture a clean background frame.
    
    Args:
        cap: OpenCV VideoCapture object
        
    Returns:
        Background frame as numpy array
        
    Raises:
        RuntimeError: If frame capture fails
    """
    print(f"Capturing background in {BACKGROUND_CAPTURE_DELAY} seconds... Stay out of the frame!")
    
    # First read might be unstable, so discard it
    ret, _ = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture initial frame")
    
    # Wait for camera to stabilize
    time.sleep(BACKGROUND_CAPTURE_DELAY)
    
    # Capture the actual background
    ret, background = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture background frame")
        
    print("Background captured successfully!")
    return background


def process_frame(frame, background):
    """
    Process a video frame to create the invisibility effect.
    
    Args:
        frame: Current video frame
        background: The captured background frame
        
    Returns:
        Processed frame with the invisibility effect applied
    """
    try:
        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for the specified color range
        mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
        
        # Clean up the mask
        mask = filter_mask(mask)
        
        # Invert the mask for the non-cloak region
        inv_mask = cv2.bitwise_not(mask)
        
        # Extract the background for the cloak region
        bg_region = cv2.bitwise_and(background, background, mask=mask)
        
        # Extract the current frame for non-cloak region
        fg_region = cv2.bitwise_and(frame, frame, mask=inv_mask)
        
        # Combine the two regions
        result = cv2.add(bg_region, fg_region)
        
        return result
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return frame  # Return original frame in case of error


def main():
    """Main function to run the invisible cloak effect."""
    try:
        with camera_capture(CAMERA_INDEX) as cap:
            # Capture clean background
            background = capture_background(cap)
            
            print("Invisible cloak activated! Press 'q' to exit.")
            
            # Main processing loop
            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame. Exiting...")
                    break
                
                # Process the frame to create invisibility effect
                result = process_frame(frame, background)
                
                # Display the result
                cv2.imshow(WINDOW_NAME, result)
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord(EXIT_KEY):
                    print("Exiting...")
                    break
                    
    except RuntimeError as e:
        print(f"Error: {str(e)}")
        return 1
    except KeyboardInterrupt:
        print("Program interrupted by user")
        return 0
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
