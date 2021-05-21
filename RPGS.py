import sys
import os
import json

from PyQt5 import uic, QtWidgets, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import webbrowser # to open pdf files
from PyQt5.QtGui import QMovie, QPixmap # for the gif


qtCreatorFile = "Main_GUI.ui"


class Ui(QtWidgets.QMainWindow):

    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi(qtCreatorFile, self)

        # REMOVE TITLE BAR
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        def moveWindow(event):
            # IF LEFT CLICK MOVE WINDOW
            if event.buttons() == Qt.LeftButton:
                self.move(self.pos() + event.globalPos() - self.dragPos)
                self.dragPos = event.globalPos()
                event.accept()

        # SET TITLE BAR
        self.title_bar.mouseMoveEvent = moveWindow

        # CLOSE
        self.btn_close.clicked.connect(lambda: self.close())

        # MINIMIZE
        self.btn_minimize.clicked.connect(lambda: self.showMinimized())

        # change the program's main window name
        self.setWindowTitle('RPGS')

        # Just for [3] label on top left
        self.label_text = ""

        # Prevent resizing
        self.setFixedSize(self.size())


        # Login page
        self.stackedWidget.setCurrentIndex(0)
        self.Username_field.returnPressed.connect(self.Login_btn.click) # enter key act as clicking the login button
        self.Password_field.returnPressed.connect(self.Login_btn.click)
        self.Login_btn.clicked.connect(self.loginfunction) # main login function

        # Choose exercise page (PATIENT) [1]
        self.Exercise_ChooseExercise.addItem("Half Arm Raise [2D]") # add the items to be selectable
        self.Exercise_ChooseExercise.addItem("Half Arm Raise [3D]")
        self.Exercise_ChooseExercise.addItem("Half Arm Raise [3D + Arduino]")
        self.Exercise_ChooseExercise.itemClicked.connect(self.Exercise_Display) # run video when exercise selected
        self.Exercise_NextBtn.clicked.connect(self.run_exercise) # run the py script depending on what user chose
        self.Logout_btn.clicked.connect(self.logout)

        # Choose exercise page (DEVELOPER) [2]
        self.Exercise_ChooseExercise_2.addItem("Half Arm Raise [2D]") # add the items to be selectable
        self.Exercise_ChooseExercise_2.addItem("Half Arm Raise [3D]")
        self.Exercise_ChooseExercise_2.addItem("Half Arm Raise [3D + Arduino]")
        self.Logout_btn_2.clicked.connect(self.logout)
        self.edit_exercise_btn.clicked.connect(self.click_to_label_variable) # go to next page while changing label text

        # Dev chosen exercise page [3]
        self.stackedWidget_edit.setCurrentIndex(0)

        with open(r"D:\STUDIES\YEAR 3 - SEM 1\ESP3902\openpose-master\build\examples\tutorial_api_python\Dev_Json\start.json") as start_json:
            start = json.load(start_json)
            for start_123 in start['start_123']:
                self.start_123_angle.setText(str(start_123["angle"]))
                self.start_123_upper.setText(str(start_123["upper"]))
                self.start_123_lower.setText(str(start_123["lower"]))
            for start_234 in start['start_234']:
                self.start_234_angle.setText(str(start_234["angle"]))
                self.start_234_upper.setText(str(start_234["upper"]))
                self.start_234_lower.setText(str(start_234["lower"]))

        with open(r"D:\STUDIES\YEAR 3 - SEM 1\ESP3902\openpose-master\build\examples\tutorial_api_python\Dev_Json\end.json") as end_json:
            end = json.load(end_json)
            for end_234 in end['end_234']:
                self.end_234_angle.setText(str(end_234["angle"]))
                self.end_234_upper.setText(str(end_234["upper"]))
                self.end_234_lower.setText(str(end_234["lower"]))
            for end_814 in end['end_814']:
                self.end_814_angle.setText(str(end_814["angle"]))
                self.end_814_upper.setText(str(end_814["upper"]))
                self.end_814_lower.setText(str(end_814["lower"]))
            for end_13 in end['end_13']:
                self.end_13_angle.setText(str(end_13["angle"]))
                self.end_13_upper.setText(str(end_13["upper"]))
                self.end_13_lower.setText(str(end_13["lower"]))
            for end_14 in end['end_14']:
                self.end_14_angle.setText(str(end_14["angle"]))
                self.end_14_upper.setText(str(end_14["upper"]))
                self.end_14_lower.setText(str(end_14["lower"]))

        with open(r"D:\STUDIES\YEAR 3 - SEM 1\ESP3902\openpose-master\build\examples\tutorial_api_python\Dev_Json\repetition.json") as rep_json:
            rep = json.load(rep_json)
            self.Number_of_reps.setValue(int(rep['repetition'][0]['repetition_number']))


        self.start_pose_edit.clicked.connect(lambda: self.stackedWidget_edit.setCurrentIndex(0))
        self.end_pose_edit.clicked.connect(lambda: self.stackedWidget_edit.setCurrentIndex(1))
        self.start_save_btn.clicked.connect(self.save_start)
        self.end_save_btn.clicked.connect(self.save_end)
        self.dev_edit_back_btn.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(2))


        # For the menu bar
        self.btn_help.clicked.connect(self.Help)


        # DONT TOUCH
        self.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # FUNCTIONS FROM HERE ONWARDS
    def loginfunction(self):
        username = self.Username_field.text()
        password = self.Password_field.text()
        self.Username_field.clear()
        self.Password_field.clear()
        if(username == "patient" and password == "patient"):
            self.stackedWidget.setCurrentIndex(1)
        elif(username== "dev" and password == "dev"):
            self.stackedWidget.setCurrentIndex(2)
        elif(username == "" and password == ""):
            return
        else:
            error_screen = QMessageBox()
            error_screen.setWindowTitle("Wrong Login")
            error_screen.setIcon(QMessageBox.Warning)
            error_screen.setText("Wrong Username or Password")
            x = error_screen.exec_()


    def Exercise_Display(self):
        if(self.Exercise_ChooseExercise.currentRow() == 0 or 1 or 2):
            movie = QMovie('instructional-video-raw.gif')
            self.Exercise_DisplayExercise.setMovie(movie)
            movie.start()


    def run_exercise(self):
        listItems = self.Exercise_ChooseExercise.selectedItems()
        if not listItems:
            return
        else:
            exercise_chosen = self.Exercise_ChooseExercise.currentItem().text()
            if(exercise_chosen == "Half Arm Raise [2D]"):
                exec(open("realsense_OP_Half_arm_raise_2D.py").read())
            elif(exercise_chosen == "Half Arm Raise [3D]"):
                exec(open("realsense_OP_Half_arm_raise_3D_V2.py").read())
            elif(exercise_chosen == "Half Arm Raise [3D + Arduino]"):
                exec(open("realsense_OP_Half_arm_raise_3D_arduino_V4(for GUI).py").read())


    def logout(self):
        logout_screen = QMessageBox()
        user_click = logout_screen.question(self,'Log Out', "Are You Sure You Want To Log Out?", logout_screen.Yes | logout_screen.No)
        if(user_click == logout_screen.Yes):
            self.stackedWidget.setCurrentIndex(0)


    def Help(self):
        # CHANGE THE PATH ACORDING TO THE CORRECT PDF
        webbrowser.open_new('instruction manual.pdf')


    def click_to_label_variable(self):
        listItems = self.Exercise_ChooseExercise_2.selectedItems()
        if not listItems:
            return
        else:
            self.label_text = self.Exercise_ChooseExercise_2.currentItem().text()
            self.stackedWidget.setCurrentIndex(3)
            self.Dev_exercise_label.setText(self.label_text)


    def save_start(self):
        save = QMessageBox()
        user_click = save.question(self,'Starting Pose Save?', "Are You Sure You Want To Save The Conditions?", save.Yes | save.No)
        if(user_click == save.Yes):
            data={}
            data['start_123'] = []
            data['start_123'].append({
                'angle': int(self.start_123_angle.text()),
                'upper': int(self.start_123_upper.text()),
                'lower': int(self.start_123_lower.text())
            })
            data['start_234'] = []
            data['start_234'].append({
                'angle': int(self.start_234_angle.text()),
                'upper': int(self.start_234_upper.text()),
                'lower': int(self.start_234_lower.text())
            })

            rep_data={}
            rep_data['repetition']=[]
            rep_data['repetition'].append({
                'repetition_number': int(self.Number_of_reps.value()),
            })

            with open(r"D:\STUDIES\YEAR 3 - SEM 1\ESP3902\openpose-master\build\examples\tutorial_api_python\Dev_Json\start.json", 'w') as outfile:
                json.dump(data, outfile)

            with open(r"D:\STUDIES\YEAR 3 - SEM 1\ESP3902\openpose-master\build\examples\tutorial_api_python\Dev_Json\repetition.json", 'w') as outfile:
                json.dump(rep_data, outfile)

            saved = QMessageBox()
            saved.setWindowTitle("Success")
            saved.setText("Starting Pose Saved")
            x = saved.exec_()


    def save_end(self):
        save = QMessageBox()
        user_click = save.question(self,'End Pose Save?', "Are You Sure You Want To Save The Conditions?", save.Yes | save.No)
        if(user_click == save.Yes):
            data={}
            data['end_234'] = []
            data['end_234'].append({
                'angle': int(self.end_234_angle.text()),
                'upper': int(self.end_234_upper.text()),
                'lower': int(self.end_234_lower.text())
            })
            data['end_814'] = []
            data['end_814'].append({
                'angle': int(self.end_814_angle.text()),
                'upper': int(self.end_814_upper.text()),
                'lower': int(self.end_814_lower.text())
            })
            data['end_13'] = []
            data['end_13'].append({
                'angle': int(self.end_13_angle.text()),
                'upper': int(self.end_13_upper.text()),
                'lower': int(self.end_13_lower.text())
            })
            data['end_14'] = []
            data['end_14'].append({
                'angle': int(self.end_14_angle.text()),
                'upper': int(self.end_14_upper.text()),
                'lower': int(self.end_14_lower.text())
            })

            rep_data={}
            rep_data['repetition']=[]
            rep_data['repetition'].append({
                'repetition_number': int(self.Number_of_reps.value()),
            })

            with open(r"D:\STUDIES\YEAR 3 - SEM 1\ESP3902\openpose-master\build\examples\tutorial_api_python\Dev_Json\end.json", 'w') as outfile:
                json.dump(data, outfile)

            with open(r"D:\STUDIES\YEAR 3 - SEM 1\ESP3902\openpose-master\build\examples\tutorial_api_python\Dev_Json\repetition.json", 'w') as outfile:
                json.dump(rep_data, outfile)

            saved = QMessageBox()
            saved.setWindowTitle("Success")
            saved.setText("End Pose Saved")
            x = saved.exec_()


    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()






app = QtWidgets.QApplication(sys.argv)
window = Ui()
# app.exec_()
app.exec_()
