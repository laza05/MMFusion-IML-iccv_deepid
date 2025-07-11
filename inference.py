import os
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.cmnext_conf import CMNeXtWithConf
from models.modal_extract import ModalitiesExtractor
from configs.cmnext_init_cfg import _C as config, update_config


class MMFusionDetector:
    def __init__(self, config_path, checkpoint_path, device=None):
        """
        Initialize the MMFusion detector.

        Args:
            config_path: Path to the yaml configuration file
            checkpoint_path: Path to the model checkpoint
            device: Device to run inference on ('cuda:0', 'cpu', etc.)
        """
        self.cfg = update_config(config, config_path)

        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Initialize models
        self.modal_extractor = ModalitiesExtractor(self.cfg.MODEL.MODALS[1:], self.cfg.MODEL.NP_WEIGHTS)
        self.model = CMNeXtWithConf(self.cfg.MODEL)

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.modal_extractor.load_state_dict(ckpt['extractor_state_dict'])

        # Set models to evaluation mode and move to device
        self.modal_extractor.to(self.device)
        self.model.to(self.device)
        self.modal_extractor.eval()
        self.model.eval()

        # Setup image transforms
        self.image_transforms_final = A.Compose([
            ToTensorV2()
        ])

    def infer(self, image_path):
        """
        Perform inference on a single image.

        Args:
            image_path: Path to the image file

        Returns:
            detection_score: Probability of manipulation (0-1)
        """
        # Load and preprocess image
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # Check if image is too large and resize if needed
        h, w = image.shape[:2]
        if h > 2048 or w > 2048:
            image = A.LongestMaxSize(max_size=2048)(image=image)['image']

        # Convert to tensor and normalize
        image_tensor = self.image_transforms_final(image=image)['image']
        image_tensor = image_tensor / 256.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            # Extract modalities
            modals = self.modal_extractor(image_tensor)

            # Normalize image for the model
            image_norm = TF.normalize(image_tensor,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

            # Prepare input and run model
            inp = [image_norm] + modals
            anomaly, confidence, detection = self.model(inp)

            # Get final detection score
            detection_score = torch.sigmoid(detection).squeeze().cpu().item()

        return detection_score

    def infer_batch(self, image_paths):
        """
        Perform inference on multiple images.

        Args:
            image_paths: List of paths to image files

        Returns:
            detection_scores: List of manipulation probabilities (0-1)
        """
        detection_scores = []

        for path in image_paths:
            score = self.infer(path)
            detection_scores.append(score)

        return detection_scores


# Example usage:
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MMFusion Inference')
    parser.add_argument('-cfg', '--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('-ckpt', '--checkpoint', type=str, required=True, help='Path to the checkpoint')
    parser.add_argument('-img', '--image', type=str, required=True, help='Path to the image')
    parser.add_argument('-gpu', '--gpu', type=int, default=0, help='GPU ID (use -1 for CPU)')

    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'

    # Initialize detector
    detector = MMFusionDetector(args.config, args.checkpoint, device)

    # Run inference
    score = detector.infer(args.image)

    print(f"Image: {args.image}")
    print(f"Manipulation detection score: {score:.4f}")
    print(f"{'MANIPULATED' if score > 0.5 else 'AUTHENTIC'} (threshold=0.5)")