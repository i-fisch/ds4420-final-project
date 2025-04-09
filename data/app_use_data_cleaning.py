"""
Author: Isabella Fisch
Date: April 4, 2025
Class: DS 4420
Assignment: Final Project
"""

import pandas as pd


def main():

    # read in dataset
    stud_apps = pd.read_excel('student_app_usage.xlsx')

    # rename columns for readability
    stud_apps.rename(columns=
                     {
                         'Amog us': 'Among Us',
                         'Asistant': 'Assistant',
                         'Chatgpt': 'ChatGPT',
                         'CLASH': 'Clash',
                     },
                     inplace=True
    )

    # drop unnecessary columns
    stud_apps.drop(columns=['no of apps ', 'Andriod '], inplace=True)

    # write to csv
    stud_apps.to_csv(
        '/Users/isabellafisch/Desktop/Woman-in-CS/DS-4420/Final-Project/student_app_usage.csv'
    )

main()
