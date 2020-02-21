# NFU-project

<ul>
  <li>yolo_object_detection_person/ : 整體系統</li>
  <ul>
    <li>face_detection/ : 臉部辨識與偵測資料夾</li>
    <ul>
      <li>FaceRec_image.py : 臉部辨識與偵測檔案</li>
    </ul>
    <li>hand_detection/ : 手勢辨識與偵測資料夾</li>
    <ul>
      <li>FaceRec_image.py : 手勢辨識與偵測檔案</li>
    </ul>
    <li>cropPerson_Video.py : 主程式</li>
    <li>DrawBox_testing.py : 測試畫方框</li>
  </ul>
  <p></p>
  <li>keras_yolo3_hand/ : 手形偵測訓練 <a href="https://github.com/qqwweee/keras-yolo3">參考連結</a></li>
  <ul>
    <li>mAP/ : mAP計算 <a href="https://github.com/Cartucho/mAP">參考連結</a></li>
    <li>trainingData/ : 訓練影像</li>
    <li>trainingAnnotation/ : 訓練Label(xml格式)</li>
    <li>annotationTxt/ : 訓練Label(txt格式, 用來訓練keras版yolo)</li>
    <li>weights/ : 已訓練好的權重</li>
    <li>train.py : 訓練手形偵測(直接執行可以不用帶參數)</li>
    <li>xml2txt : 將xml格式Label轉換成txt格式Label</li>
  </ul>
  <p></p>
  <li>FaceRecognition/ : 臉部辨識訓練</li>
  <ul>
    <li>FaceModelC.py : 訓練臉部辨識模型</li>
    <li>FaceRec_image.py : 臉部辨識預測</li>
    <li>img2npy.py : 將影像(從Face/data/)轉換成npy(Data.npy, Label.npy)</li>
  </ul>
  <p></p>
  <li>Hand/ : 手勢辨識訓練</li>
  <ul>
    <li>HandModel.py : 訓練手勢辨識模型</li>
    <li>predictModel.py : 手勢辨識預測</li>
    <li>img2npy.py : 將影像(從CaptureHandPackage/HandData_aug/)轉換成npy(Data.npy, Label.npy)</li>
  </ul>
  <p></p>
  <li>CaptureHandPackage/ : 擷取手形影像</li>
  <ul>
    <li>CatchPIC_fromfile.py : 從檔案讀取影片</li>
    <li>XMLaug.py : 影像增強資料集(HandData/到HandData_aug/) <a href="https://github.com/aleju/imgaug">參考連結</a></li>
  </ul>
  <p></p>
  <li>CaptureFacePackage/ : 擷取臉形影像</li>
  <ul>
    <li>CatchPIC_fromfile.py : 從檔案讀取影片</li>
  </ul>
  <p></p>
  <li>labelImg-master/ : 手動製作物件偵測labal程式 <a href="https://github.com/tzutalin/labelImg">參考連結</a></li>
  <ul>
    <li>screenShot.py : 從影片內擷取整張圖片</li>
  </ul>
</ul>






