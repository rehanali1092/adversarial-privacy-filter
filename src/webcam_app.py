"""
webcam_app.py
Author: Rehan Ali
Real-time adversarial privacy filter for webcam
"""

import cv2
import torch
import numpy as np
import mediapipe as mp
import torchvision.transforms as transforms
from PIL import Image
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.model import FaceRecognitionModel
from src.attack import FGSMAttack


class AdversarialPrivacyFilter:
    """
    Real-time adversarial privacy filter.
    Captures webcam feed, detects faces,
    applies FGSM perturbation to protect identity.
    """

    def __init__(self, model_path, num_classes=62,
                 epsilon=0.05):

        # Device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Running on: {self.device}")

        # Load model
        print("Loading model...")
        self.model = FaceRecognitionModel(
            num_classes=num_classes
        )
        self.model.load_state_dict(
            torch.load(model_path,
                      map_location=self.device)
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded!")

        # Attack
        self.attack  = FGSMAttack(self.model, epsilon)
        self.epsilon = epsilon

        # Face detector
        self.mp_face    = mp.solutions.face_detection
        self.detector   = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.7
        )

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

        # State
        self.protection_on = True
        self.frame_count   = 0
        self.fps           = 0
        self.fps_time      = time.time()

        print("✅ Adversarial Privacy Filter ready!")
        print()
        print("Controls:")
        print("  SPACE → Toggle protection ON/OFF")
        print("  Q     → Quit")

    def detect_face(self, frame):
        """Detect faces and return bounding boxes"""
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.process(rgb)
        boxes  = []

        if result.detections:
            h, w = frame.shape[:2]
            for det in result.detections:
                bb  = det.location_data.relative_bounding_box
                x1  = max(0, int(bb.xmin * w))
                y1  = max(0, int(bb.ymin * h))
                x2  = min(w, int((bb.xmin + bb.width) * w))
                y2  = min(h, int((bb.ymin + bb.height) * h))
                if (x2 - x1) > 30 and (y2 - y1) > 30:
                    boxes.append((x1, y1, x2, y2))
        return boxes

    def protect_face(self, frame, box):
        """Apply adversarial perturbation to face region"""
        x1, y1, x2, y2 = box
        face_crop = frame[y1:y2, x1:x2].copy()

        if face_crop.size == 0:
            return frame

        # To PIL and transform
        face_pil    = Image.fromarray(
            cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        )
        face_tensor = self.transform(face_pil).unsqueeze(0)
        face_tensor = face_tensor.to(self.device)

        # Dummy label for attack
        dummy_label = torch.zeros(1, dtype=torch.long)
        dummy_label = dummy_label.to(self.device)

        # Generate perturbation
        adv_tensor, perturbation = self.attack.attack(
            face_tensor, dummy_label
        )

        # Convert perturbation to pixel space
        pert_np = perturbation.squeeze(0)
        pert_np = pert_np.permute(1, 2, 0).cpu().detach().numpy()
        pert_np = (pert_np * 255).clip(-30, 30).astype(np.int16)

        # Resize to face crop size
        pert_resized = cv2.resize(
            pert_np.astype(np.float32),
            (face_crop.shape[1], face_crop.shape[0])
        ).astype(np.int16)

        # Apply perturbation
        pert_bgr       = pert_resized[:, :, ::-1]
        protected_face = face_crop.astype(np.int16) + pert_bgr
        protected_face = np.clip(protected_face,
                                 0, 255).astype(np.uint8)

        # Put back in frame
        result_frame              = frame.copy()
        result_frame[y1:y2, x1:x2] = protected_face
        return result_frame

    def draw_ui(self, frame, boxes):
        """Draw status overlay on frame"""
        h, w = frame.shape[:2]

        for (x1, y1, x2, y2) in boxes:
            color = (0, 255, 0) if self.protection_on \
                    else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                         color, 2)
            label = "PROTECTED" if self.protection_on \
                    else "UNPROTECTED"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, color, 2)

        # Status bar
        status = "ON" if self.protection_on else "OFF"
        color  = (0, 255, 0) if self.protection_on \
                 else (0, 0, 255)
        cv2.putText(frame,
                   f"Protection: {status} | "
                   f"FPS: {self.fps:.1f} | "
                   f"e={self.epsilon:.2f}",
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, color, 2)
        cv2.putText(frame,
                   "SPACE: Toggle | Q: Quit",
                   (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (200, 200, 200), 1)
        return frame

    def run(self, camera=0):
        """Main application loop"""
        cap = cv2.VideoCapture(camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Error: Cannot open webcam!")
            return

        print("✅ Webcam opened!")
        print("Running... Press Q to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1

            # FPS calculation
            if self.frame_count % 30 == 0:
                elapsed    = time.time() - self.fps_time
                self.fps   = 30 / elapsed
                self.fps_time = time.time()

            # Detect faces
            boxes = self.detect_face(frame)

            # Apply protection
            if self.protection_on and boxes:
                for box in boxes:
                    frame = self.protect_face(frame, box)

            # Draw UI
            frame = self.draw_ui(frame, boxes)

            cv2.imshow('Adversarial Privacy Filter', frame)

            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                self.protection_on = not self.protection_on
                status = "ON" if self.protection_on else "OFF"
                print(f"Protection: {status}")

        cap.release()
        cv2.destroyAllWindows()
        print("Closed.")


if __name__ == "__main__":
    MODEL_PATH = "models/best_model.pth"

    app = AdversarialPrivacyFilter(
        model_path=MODEL_PATH,
        num_classes=62,
        epsilon=0.05
    )
    app.run(camera=0)
