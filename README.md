# NFU-project


yolo_object_detection_person : 整體系統
<ul>
  <li>keras_yolo3_hand : 手形偵測訓練</li>
  <ul>
    <li>trainingData/ : 訓練影像</li>
    <li>trainingAnnotation/ : 訓練Label(xml格式)</li>
    <li>annotationTxt/ : 訓練Label(txt格式, 用來訓練keras版yolo)</li>
    <li>weights/ : 已訓練好的權重</li>
    <li>train.py : 訓練手形偵測(直接執行可以不用帶參數)</li>
    <li>xml2txt : 將xml格式Label轉換成txt格式Label</li>
  </ul>
  <li>FaceRecognition : 臉部辨識訓練</li>
  <ul>
    <li>FaceModelC.py : 訓練臉部辨識模型</li>
    <li>FaceRec_image.py : 臉部辨識預測</li>
    <li>img2npy.py : 將影像(從Face/data/)轉換成npy(Data.npy, Label.npy)</li>
  </ul>
  <li>Hand : 手勢辨識訓練</li>
  <ul>
    <li>HandModel.py : 訓練手勢辨識模型</li>
    <li>predictModel.py : 手勢辨識預測</li>
    <li>img2npy.py : 將影像(從CaptureHandPackage/HandData_aug/)轉換成npy(Data.npy, Label.npy)</li>
  </ul>
  <li>CaptureHandPackage : 擷取手形影像</li>
  <ul>
    <li>CatchPIC_fromfile.py : 從檔案讀取影片</li>
    <li>XMLaug.py : 影像增強資料集(HandData/到HandData_aug/)
  </ul>
  <li>CaptureFacePackage : 擷取臉形影像</li>
  <ul>
    <li>CatchPIC_fromfile.py : 從檔案讀取影片</li>
  </ul>
</ul>






