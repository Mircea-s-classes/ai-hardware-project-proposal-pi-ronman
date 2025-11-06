# University of Virginia
## Department of Electrical and Computer Engineering

**Course:** ECE 4332 / ECE 6332 — AI Hardware Design and Implementation  
**Semester:** Fall 2025  
**Proposal Deadline:** November 5, 2025 — 11:59 PM  
**Submission:** Upload to Canvas (PDF) and to GitHub (`/docs` folder)

---

# AI Hardware Project Proposal Template

## 1. Project Title
Name of the Team: "Pi"ronman

List of students in the team: Jordan Capelle, Mohomad Sabur, Karl Garman

Provide a clear and concise title for your project: 
Eyes in the Sky: AI Bird Detection for Drone Safety 

## 2. Platform Selection
Select one platform category and justify your choice:
Our project requires real-time image processing on a small unmanned aircraft system (sUAS), where low latency and onboard decision-making are critical for collision avoidance. Running YOLO on a Raspberry Pi 5 with an AI Hat allows the model to detect and classify birds locally without relying on cloud connectivity. This reduces transmission delays, ensures faster response times, and supports autonomous operation even in areas without a network connection.

**Undergraduates:** Edge-AI, TinyML, or Neuromorphic platforms  
**Graduates:** open-source AI accelerators (Ztachip, VTA, Gemmini, VeriGOOD-ML, NVDLA) or any of the above 

## 3. Problem Definition
Describe the AI or hardware design problem you aim to address and its relevance to AI hardware (e.g., efficiency, latency, scalability).

Our project addresses the challenge of enabling real-time bird detection and avoidance on a resource-limited onboard system. The key hardware problem is achieving low-latency image inference while maintaining power efficiency and flight-ready performance on a compact embedded platform (Raspberry Pi 5 + AI Hat).

This problem is directly relevant to AI hardware design because it tests how well edge processors can support computationally intensive neural networks like YOLO under strict constraints of size, weight, power, and response time. Solving it demonstrates how optimized embedded AI systems can extend autonomy to unmanned aerial vehicles and other real-time safety applications.

## 4. Technical Objectives
List 3–5 measurable objectives with quantitative targets when possible.

1) Real-Time Detection:
Achieve ≥15 FPS inference speed on the Raspberry Pi 5 with AI Hat during live video capture.

2) Detection Accuracy:
Attain ≥85% precision and ≥80% recall in classifying birds vs. non-birds in test footage.

3) Latency Minimization:
Maintain <200 ms end-to-end response time from image capture to detection output.

4) System Reliability:
Successfully detect and log bird encounters in >95% of field test flights without system crash or frame drop.

## 5. Methodology
Describe your planned approach: hardware setup, software tools, model design, performance metrics, and validation strategy.

Hardware Setup
- Compute: Raspberry Pi 5 (8GB) + AI Hat/accelerator (Hailo)
- Sensing: CSI/USB camera
- Thermals: Heatsink + fan; temperature monitor during runs

Software Tools
- OS & Runtime: Ubuntu on Pi, Python 3.13 venv
- Libraries: Ultralytics YOLO (yolo11n/s)
- Data: Public bird datasets
  
Model Design
- Baseline: YOLO11n/s pretrained on COCO, class-filtered to “bird”
- Fine-tuning: Short transfer-learning on bird/non-bird + hard negatives (kites, planes, clouds)
- Outputs: (a) detection + confidence, (b) approximate bearing from image coordinates, (c) rate-of-change for time-to-collision proxy

Performance Metrics
- Speed: FPS (target ≥15) and end-to-end latency capture→decision (<200 ms)
- Accuracy: Precision/Recall, mAP@0.5 for “bird”; confusion matrix with common distractors
- Robustness: Performance vs. distance, lighting, motion blur, occlusion; missed/false alert rates
- Reliability: % runs without crash or >1-second frame stall (target >95%)

Validation Strategy
1) Dataset Evaluation:
Use public and team-collected bird videos. Annotate samples to measure precision, recall, and mAP@0.5 for the “bird” class.
2) Performance Testing:
Run detection on recorded videos to measure FPS and latency (<200 ms) on the Raspberry Pi 5 + AI Hat.
3) Robustness Checks:
Apply synthetic variations (lighting, blur, scale) to test detection stability under different conditions.
4) Live Camera Validation:
Use a USB/webcam feed to verify real-time inference and overlay performance on a live image stream.
5) System Efficiency:
Log CPU/GPU usage, temperature, and power draw, ensuring consistent operation during extended runs.

## 6. Expected Deliverables
List tangible outputs: working demo, GitHub repository, documentation, presentation slides, and final report.
- Working Demo: Real-time bird detection using YOLO on the Raspberry Pi 5 with live video feed and FPS logging.
- GitHub Repository: Full project code, setup guide, and validation scripts for reproducibility.
- Documentation: Step-by-step hardware and software setup instructions with troubleshooting notes.
- Presentation Slides: Summarizing design, results, and performance analysis for final presentation.
- Final Report: Comprehensive summary of methods, testing outcomes, and recommendations for future on-drone implementation.

## 7. Team Responsibilities
List each member’s main role.

| Name | Role | Responsibilities |
|------|------|------------------|
| Jordan Capelle | Systems Integration Engineer | Hardware Setup & YOLO integration |
| Karl Garman | Research & Analysis Engineer | Literature Review & Documentation |
| Mohomad Sabur | Lead Software Engineer | AI Model Coding & Optimization |

## 8. Timeline and Milestones
Provide expected milestones:

- November 8th: Harware Setup
- November 19th: Code and Train Complete
- December 10th: Test and Validate
- December 17th: Presentation and Paper Complete

## 9. Resources Required
List special hardware, datasets, or compute access needed.

Hardware:
- Raspberry Pi 5 (8 GB)
- AI accelerator
- USB or CSI camera module
- Power supply, heatsink, and fan for sustained inference

Datasets:
- Public bird image/video datasets

Compute Access:
- Local desktop PCs for model training, testing, and data preprocessing

## 10. References
Include relevant papers, repositories, and documentation.

- Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv:1804.02767.
- Bochkovskiy, A., Wang, C.-Y., & Liao, H.-Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv:2004.10934.
- Ultralytics. (2024). YOLOv8 / YOLO11 Documentation. Retrieved from https://docs.ultralytics.com
- BirdCLEF Dataset. (2023). Bird Species Recognition Challenge Dataset. Retrieved from https://www.kaggle.com/competitions/birdclef-2023
- Raspberry Pi Foundation. (2024). Raspberry Pi 5 Documentation. Retrieved from https://www.raspberrypi.com/documentation/
- Google Coral. (2023). Edge TPU Python API. Retrieved from https://coral.ai/docs/
