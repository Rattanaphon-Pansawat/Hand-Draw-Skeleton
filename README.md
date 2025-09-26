# Hand Idea Skeleton

สร้างโครงกระดูกมือ (skeleton) ตาม **Idea ของ Rattanaphon-pansawat**  
แนวคิด: วาดเส้นตัดที่ตำแหน่งข้อ → ปลายเส้นชนขอบมือ = จุดฟ้า → เอากึ่งกลาง = จุดแดง → เชื่อมจุดแดงเป็นโครงกระดูก  
**ไม่ใช้ MediaPipe** ใช้เพียง OpenCV + NumPy

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt```


## Run

python hand_skeleton.py

* กด q ออกจากโปรแกรม
* กด s เพื่อบันทึก snapshot → เก็บใน output_samples/
