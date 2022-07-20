import os
from pathlib import Path
import sys
import PySimpleGUI as sg
from prediction import fungus_Prediction

sg.theme_background_color("#F0F0F0")


col1=[[sg.Image('home_logo.png',background_color='#F0F0F0')],[sg.Text("",background_color='#F0F0F0', key="-Output-", font=40)]]
col2=[[sg.Input(disabled=True, key='-INPUT-',visible=False), sg.Button('Select a picture', size=(30,3), button_color=("green",'#E8FFCD'))],]
layout = [[sg.Column(col1, element_justification='c'),
          sg.Column(col2,element_justification='r')] ]

window = sg.Window('SoiLab',icon='icon.ico',size=(800,450), grab_anywhere=True,).Layout(layout)

while True:             # Event Loop
    event, values = window.Read()
    if event in  (None, 'Exit'):
        break
    elif event == "Select a picture":
        file = sg.popup_get_file('Select a picture to be classified', text_color="black", button_color=("green",'#E8FFCD'),background_color='#F0F0F0', file_types=[("PNG Files", '*.png')])
        if(Path(file).is_file()):
            fungus = fungus_Prediction(file)
            result = "Fungus on a photo is {}".format(fungus)
            window["-Output-"].update(result, text_color="black")
window.Close()

