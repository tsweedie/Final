## Final
#Face Detection, Feature Extraction & SVM Modelling(+ Testing)
  myFeatures.py
    -Inputs:haarcascade_frontalface_default.xml and haarcascade_eye.xml
    -Viola Jones 
    -HOG Gradients
    -HOG(OpenCV)
  mineData.py
    -Inputs: CK+ datasets(Cleaned up)
    -Extract CK+ dataset from folders
    -Run Viola Jones on dataset
    -Run HOG(OpenCV) on dataset(after viola)
    -Outputs: dataExcel.xlsx, dataCsv.csv and folders for Viola Jones images
  modellingSVM.py
    -Inputs: dataCsv.csv
    -Grid Search
    -Train SVM
    -Test SVM
    -Outputs:finalized_model.sav
#GUI for Live Stream Classification
  guiAHED.py
    -Inputs:finalized_model.sav, emojis,lense.png and
            userinterface.ui(XML format PyQt Designer)
    -Outputs: GUI
  
#Jupyter Notebook for 'modellingSVM.py'
  traintestGUI.ipynb
    -same as 'modellingSVM.py'
