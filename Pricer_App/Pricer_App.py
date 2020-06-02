# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:02:02 2020

@author: Wladyslaw Eysymontt
"""

from tkinter import ttk
import tkinter as tk
from tkinter.ttk import Progressbar,Entry
from tkinter import Tk,DISABLED,StringVar,Label,W,E,Text,HORIZONTAL,Radiobutton
import tkinter.font as tkFont
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import pandas as pd
import numpy as np
import ANN_predictions
os.chdir('C:\Github_repositories\AI-Based_Pricer\Pricer_App')


class Product:
    def __init__(self):
        window = Tk()
        self.wind = window
        self.wind.geometry("1060x670") 
        self.wind.title('Artificial intelligence based pricer')
        self.wind.lift()
        self.wind.resizable(width=False, height=False)
        self.wind.iconbitmap("../Files/Images/app_icon.ico")
        
        background_color = '#526b70'
        background_color2 = '#e4e8eb'
        background_color3 = '#e4e8eb'
        background_color4 = '#c4cfd2'
        self.text_color = 'white'
        self.incorrect_color = '#fa7878'
        text_color2 = '#1b2325'
        text_color3 = '#3a4d50'
        
        font = {'family' : 'Verdana',
        'weight' : 'normal',
        'size'   : 10}
        matplotlib.rc('font', **font)
        
        self.wind.option_add("*TCombobox*Listbox*Background", background_color3)
        
        img = Image.open('../Files/Images/background_image.jpg')
        img = ImageTk.PhotoImage(img)
        background = tk.Label(self.wind, image=img, bd=0)
        background.grid(row = 0, column = 0, rowspan = 8, columnspan = 5)
        background.configure(background=background_color)
        
        img2 = Image.open('../Files/Images/background_image2.jpg')
        img2 = ImageTk.PhotoImage(img2)
        background2 = tk.Label(self.wind, image=img2, bd=0)
        background2.grid(row = 9, column = 0, rowspan = 10, columnspan = 10)
        background2.configure(background=background_color4)
    
        self.plots_data = ''
        
    
        ########## LEFT TOP SIDE ##############################
        
        fontStyleTitle = tkFont.Font(family='Times New Roman (Times)', size=12, weight='bold')
        fontStyleText = tkFont.Font(family='Arial', size=10, weight='bold')
        
        Label1 = Label(background, text = '    ')
        Label1.grid(row = 0, column = 0)
        Label1.configure(background=background_color)
        Label2 = Label(background, text = 'ANN CONFIGURATION', fg=self.text_color, font=fontStyleTitle)
        Label2.grid(row = 1, column = 1, padx = 3, columnspan = 2, sticky = W)
        Label2.configure(background=background_color)
        
        self.Label3 = Label(background, text = 'Model: ', fg=self.text_color, font=fontStyleText)
        self.Label3.grid(row = 2, column = 1, pady = 4, sticky = W)
        self.Label3.configure(background=background_color)
        self.last_filter_mode = ''
        self.filter_mode = StringVar();
        self.filter_mode.set('')
        self.model = ttk.Combobox(background, textvariable = self.filter_mode, state='readonly', values=getModels())
        self.model.grid(row = 2, column = 2)
        
        self.Label4 = Label(background, text = 'Test set: ', fg=self.text_color, font=fontStyleText)
        self.Label4.grid(row = 3, column = 1, sticky = W)
        self.Label4.configure(background=background_color)
        self.last_filter_mode2 = ''
        self.filter_mode2 = StringVar();
        self.filter_mode2.set('')
        self.test_set = ttk.Combobox(background, textvariable = self.filter_mode2, state='readonly', values=[''])
        self.test_set.grid(row = 3, column = 2)
        
        Label5 = Label(background, text = '')
        Label5.grid(row = 4, column = 1)
        Label5.configure(background=background_color)
        Label6 = Label(background, text = 'DATA CONFIGURATION', font = fontStyleTitle, fg=self.text_color)
        Label6.grid(row = 5, column = 1, padx = 3, columnspan = 2, sticky = W)
        Label6.configure(background=background_color)
        
        self.Label7 = Label(background, text = 'Shop ID: ', fg=self.text_color, font=fontStyleText)
        self.Label7.grid(row = 6, column = 1, pady = 4, sticky = W)
        self.Label7.configure(background=background_color)
        self.last_filter_mode3 = ''
        self.filter_mode3 = StringVar();
        self.filter_mode3.set('')
        self.shop = ttk.Combobox(background, textvariable = self.filter_mode3, state='readonly', values=[''])
        self.shop.grid(row = 6, column = 2)
        
        self.Label8 = Label(background, text = 'Item ID: ', fg=self.text_color, font=fontStyleText)
        self.Label8.grid(row = 7, column = 1, sticky = W)
        self.Label8.configure(background=background_color)
        self.item = ttk.Combobox(background, state='readonly', values=[''])
        self.item.grid(row = 7, column = 2)
        
        
        ########## CENTER TOP SIDE ############################
        
        Label9 = Label(background, text = 'ANN MODEL PERFORMANCE', font=fontStyleTitle, fg=self.text_color)
        Label9.grid(row = 1, column = 3, padx = 40, sticky = W)
        Label9.configure(background=background_color)
        
        self.performance = Text(background, height=8, width=50)
        self.performance.grid(row = 2, column = 3, padx = 40, rowspan = 6)
        temporalText = ''
        self.performance.configure(background=background_color3)
        self.performance.insert(tk.END, temporalText)
        self.performance.config(state=DISABLED)
        
        self.progress = Progressbar(background, style='TProgressbar', orient = HORIZONTAL, length = 100, mode = 'determinate')
        self.progress.grid(row = 8, column = 3, padx = 40, sticky = W+E)
        
        Button1 = tk.Button(background, bg=background_color2, fg=text_color2, text = 'Calculate performance', command=lambda:ANN_predictions.calculatePerformance(self))
        Button1.grid(row = 8, column = 1, padx = 50, pady = 10, columnspan = 2, sticky = W+E)
        

        ########## RIGHT TOP SIDE #############################
        
        Label10 = Label(background, text = '        ')
        Label10.grid(row = 0, column = 6)
        Label10.configure(background=background_color)
        Label11 = Label(background, text = "PREDICTION'S CONFIGURATION", font=fontStyleTitle, fg=self.text_color)
        Label11.grid(row = 1, column = 4, padx = 3, columnspan = 2, sticky = W)
        Label11.configure(background=background_color)
        
        self.Label12 = Label(background, text = 'Precision: ', fg=self.text_color, font=fontStyleText)
        self.Label12.grid(row = 2, column = 4, pady = 4, sticky = W)
        self.Label12.configure(background=background_color)
        self.precision = Entry(background)
        self.precision.focus()
        self.precision.grid(row = 2, column = 5)
        self.precision.configure(background=self.text_color)
        
        self.Label13 = Label(background, text = 'Max. price multiplicator: ', fg=self.text_color, font=fontStyleText)
        self.Label13.grid(row = 3, column = 4, sticky = W)
        self.Label13.configure(background=background_color)
        self.max_price_multiplicator = Entry(background)
        self.max_price_multiplicator.grid(row = 3, column = 5)
        self.max_price_multiplicator.configure(background=self.text_color)
        
        self.Label14 = Label(background, text = 'Delta multiplicator: ', fg=self.text_color, font=fontStyleText)
        self.Label14.grid(row = 4, column = 4, pady = 4, sticky = W)
        self.Label14.configure(background=background_color)
        self.delta_multiplicator = Entry(background)
        self.delta_multiplicator.grid(row = 4, column = 5)
        self.delta_multiplicator.configure(background=self.text_color)
        
        self.Label15 = Label(background, text = 'Item cost: ', fg=self.text_color, font=fontStyleText)
        self.Label15.grid(row = 5, column = 4, sticky = W)
        self.Label15.configure(background=background_color)
        self.item_cost = Entry(background)
        self.item_cost.grid(row = 5, column = 5)
        self.item_cost.configure(background=self.text_color)
        self.selectedBtn = StringVar()
        Radiobutton(background, text = "absolute", variable = self.selectedBtn, value = "1", indicator = 0, fg=text_color2, background = "light blue", activebackground=background_color2, activeforeground='black').grid(row = 6, column = 4, sticky = E)
        Radiobutton(background, text = "mean price multiplicator", variable = self.selectedBtn, value = "2", indicator = 0, fg=text_color2, background = "light blue", activebackground=background_color2, activeforeground='black').grid(row = 6, column = 5, pady = 4, sticky = W)
        
        self.Label16 = Label(background, text = 'Fixed costs: ', fg=self.text_color, font=fontStyleText)
        self.Label16.grid(row = 7, column = 4, sticky = W)
        self.Label16.configure(background=background_color)
        self.fixed_costs = Entry(background)
        self.fixed_costs.grid(row = 7, column = 5)
        self.fixed_costs.configure(background=self.text_color)
        
        Button2 = tk.Button(background, fg=text_color2, bg=background_color2, text = 'Calculate predictions', command=lambda:ANN_predictions.calculatePredictions(self, background_color4))
        Button2.grid(row = 8, column = 4, padx = 80, pady = 10, columnspan = 2, sticky = W+E)
        
        Label17 = Label(background, text = '                                                                                                                                                                                                                                                               ')
        Label17.grid(row = 0, column = 6, sticky = W)
        Label17.configure(background=background_color)

        
        ########## LEFT BOTTOM SIDE ###########################
        reversed_space_between_labels = '       '
        
        Label18 = Label(background, text = '        ')
        Label18.grid(row = 9, column = 1)
        Label18.configure(background=background_color)
        
        Label19 = Label(background2, text = '             ')
        Label19.grid(row = 0, column = 0, sticky = W)
        Label19.configure(background=background_color4)
        
        Label20 = Label(background2, text = reversed_space_between_labels)
        Label20.grid(row = 1, column = 2)
        Label20.configure(background=background_color4)
        
        Label21 = Label(background2, text = 'Algorithm: ', fg=text_color3, font=fontStyleText)
        Label21.grid(row = 1, column = 3, sticky = W)
        Label21.configure(background=background_color4)
        
        self.algorithm1 = ttk.Combobox(background2, state='readonly', values=["Optimized","Pure ANN"])
        self.algorithm1.current(0)
        self.algorithm1.bind("<<ComboboxSelected>>", lambda x: changeLeftPlot(self))
        self.algorithm1.grid(row = 1, column = 4, sticky = W)
        
        Label22 = Label(background2, text = '    ')
        Label22.grid(row = 1, column = 5, sticky = W)
        Label22.configure(background=background_color4)
        
        Label23 = Label(background2, text = 'Plot: ', fg=text_color3, font=fontStyleText)
        Label23.grid(row = 1, column = 6, sticky = E)
        Label23.configure(background=background_color4)
        
        self.plot_type1 = ttk.Combobox(background2, state='readonly', values=["Benefits","Sales"])
        self.plot_type1.current(0)
        self.plot_type1.bind("<<ComboboxSelected>>", lambda x: changeLeftPlot(self))
        self.plot_type1.grid(row = 1, column = 7, sticky = E)
        
        Label24 = Label(background2, text = reversed_space_between_labels)
        Label24.grid(row = 1, column = 8)
        Label24.configure(background=background_color4)
        
        self.fig = Figure(figsize=(4.5, 3.1), dpi=100)
        self.fig.set_facecolor(background_color4)
        self.fig.subplots_adjust(left = 0.2, bottom = 0.14)
        self.plot1 = self.fig.add_subplot(111)
        self.plot1.set_title('Benefits / Price (optimized algorithm)', size=12)
        self.plot1.set_ylabel('Benefits', fontsize = 12)
        self.plot1.set_xlabel('Price', fontsize = 12)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=background2)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row = 2, column = 2, columnspan = 7)
        
        
        ########## CENTRAL BOTTOM SIDE #########################
        Label25 = Label(background2, text = '   ')
        Label25.grid(row = 1, column = 9)
        Label25.configure(background=background_color4)
        
        
        ########## RIGHT BOTTOM SIDE ###########################
        
        Label26 = Label(background2, text = reversed_space_between_labels)
        Label26.grid(row = 1, column = 10, sticky = W)
        Label26.configure(background=background_color4)
        
        Label27 = Label(background2, text = 'Algorithm: ', fg=text_color3, font=fontStyleText)
        Label27.grid(row = 1, column = 11, sticky = W)
        Label27.configure(background=background_color4)
        
        self.algorithm2 = ttk.Combobox(background2, state='readonly', values=["Optimized","Pure ANN"])
        self.algorithm2.current(0)
        self.algorithm2.bind("<<ComboboxSelected>>", lambda x: changeRightPlot(self))
        self.algorithm2.grid(row = 1, column = 12, sticky = W)
        
        Label28 = Label(background2, text = '    ')
        Label28.grid(row = 1, column = 13)
        Label28.configure(background=background_color4)
        
        Label29 = Label(background2, text = 'Plot: ', fg=text_color3, font=fontStyleText)
        Label29.grid(row = 1, column = 14, sticky = E)
        Label29.configure(background=background_color4)
        
        self.plot_type2 = ttk.Combobox(background2, state='readonly', values=["Benefits","Sales"])
        self.plot_type2.current(1)
        self.plot_type2.bind("<<ComboboxSelected>>", lambda x: changeRightPlot(self))
        self.plot_type2.grid(row = 1, column = 15, sticky = E)
        
        Label30 = Label(background2, text = reversed_space_between_labels)
        Label30.grid(row = 1, column = 16, sticky = W)
        Label30.configure(background=background_color4)
        
        self.fig2 = Figure(figsize=(4.5, 3.1), dpi=100)
        self.fig2.set_facecolor(background_color4)
        self.fig2.subplots_adjust(left = 0.2, bottom = 0.14)
        self.plot2 = self.fig2.add_subplot(111)
        self.plot2.set_title('Sales / Price (optimized algorithm)', size=12)
        self.plot2.set_ylabel('Sales', fontsize = 12)
        self.plot2.set_xlabel('Price', fontsize = 12)
        
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=background2)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().grid(row = 2, column = 10, columnspan = 7)

        Label31 = Label(background2, text = '                                                                                                                                                                                                                                                                                      ')
        Label31.grid(row = 1, column = 17, sticky = W)
        Label31.configure(background=background_color4)
        
        text1 = 'Optimal price: 0'
        self.text1 = Text(background2, height=1, width=25)
        self.text1.grid(row = 3, column = 3, sticky = W, columnspan = 2)
        self.text1.configure(background=background_color4)
        self.text1.insert(tk.END, text1)
        self.text1.config(state=DISABLED)
        
        text2 = 'Corr. benefits: 0'
        self.text2 = Text(background2, height=1, width=25)
        self.text2.grid(row = 3, column = 5, sticky = E, columnspan = 3)
        self.text2.configure(background=background_color4)
        self.text2.insert(tk.END, text2)
        self.text2.config(state=DISABLED)
        
        text3 = 'Mean price: 0'
        self.text3 = Text(background2, height=1, width=25)
        self.text3.grid(row = 3, column = 11, sticky = W, columnspan = 2)
        self.text3.configure(background=background_color4)
        self.text3.insert(tk.END, text3)
        self.text3.config(state=DISABLED)
        
        text4 = 'Corr. sales: 0'
        self.text4 = Text(background2, height=1, width=25)
        self.text4.grid(row = 3, column = 13, sticky = E, columnspan = 3)
        self.text4.configure(background=background_color4)
        self.text4.insert(tk.END, text4)
        self.text4.config(state=DISABLED)
        
        
        ########## BOTTOM MARGIN ###############################
        
        Label32 = Label(background2, text = '  ')
        Label32.grid(row = 4, column = 1, pady = 6, columnspan = 15)
        Label32.configure(background=background_color4)


    def update(self):
        filter_mode = self.filter_mode.get()
        filter_mode2 = self.filter_mode2.get()
        filter_mode3 = self.filter_mode3.get()

        if filter_mode != self.last_filter_mode: 
            self.test_set['values'] = getTestSets(self.model)
            self.last_filter_mode = filter_mode
        
        if filter_mode2 != self.last_filter_mode2: 
            self.shop['values'] = getShops(self.test_set)
            self.last_filter_mode2 = filter_mode2

        if filter_mode3 != self.last_filter_mode3:
            self.item['values'] = getItems(self.test_set,self.shop)
            self.last_filter_mode3 = filter_mode3
        
        self.wind.after(100, self.update)


    def main(self):
        self.update()
        self.wind.mainloop()



def getModels():
    entries = os.listdir('../Files/Trained_ANNs')
    models = []
    for entry in entries:
        splitted_entry = entry.split(".")[0]
        models.append(splitted_entry)
    return models


def getTestSets(model):
    selected_model = model.get()
    if selected_model != '':
        if 'standarized' in selected_model:
            selected_model = selected_model.split('_')[1]
            modelTimeWindow = selected_model
        else:
            modelTimeWindow = selected_model[0]
    else:
        modelTimeWindow = 0
    entries = os.listdir('../Files/Completely_processed_data')
    test_sets = []
    for entry in entries:
        if 'test' in entry:
            splitted_entry = entry.split('test_')[1].split('.')[0]
            datasetTimeWindow = entry.split('_')[3]
            if datasetTimeWindow == modelTimeWindow:
                test_sets.append(splitted_entry)
    return test_sets


def getShops(test_set):
    selected_set = test_set.get()
    if selected_set != '':
        test_df = pd.read_csv(("../Files/Completely_processed_data/shaped_input_test_{}.csv").format(selected_set))
        shops = np.sort(test_df['shop_id'].unique()).tolist()
    else:
        shops = ['']
    return shops


def getItems(test_set,shop):
    selected_set = test_set.get()
    selected_shop = shop.get()
    if selected_set != '' and selected_shop != '':
        test_df = pd.read_csv(("../Files/Completely_processed_data/shaped_input_test_{}.csv").format(selected_set))
        filtered_test_df = test_df['shop_id']==int(selected_shop)
        filtered_test_df = test_df[filtered_test_df]
        items = np.sort(filtered_test_df['item_id'].unique()).tolist()
    else:
        items = ['']
    return items


def changeLeftPlot(self):
    algorithm = self.algorithm1.get()
    plot_type = self.plot_type1.get()
    
    if self.plots_data != '':
        self.plot1.clear()
        self.text1.config(state='normal')
        self.text1.delete('1.0', tk.END)
        self.text2.config(state='normal')
        self.text2.delete('1.0', tk.END)
        if algorithm == 'Optimized' and plot_type == 'Benefits':
            self.plot1.plot(self.plots_data[0],self.plots_data[4])
            self.plot1.set_title('Benefits / Price (optimized algorithm)', size=12)
            self.plot1.set_ylabel('Benefits', fontsize = 12)
            self.text1.insert(tk.END, (('Optimal price: {}').format(self.plots_data[9])))
            self.text2.insert(tk.END, (('Corr. benefits: {}').format(self.plots_data[10])))
        elif algorithm == 'Optimized' and plot_type == 'Sales':
            self.plot1.plot(self.plots_data[0],self.plots_data[2])
            self.plot1.set_title('Sales / Price (optimized algorithm)', size=12)
            self.plot1.set_ylabel('Sales', fontsize = 12)
            self.text1.insert(tk.END, (('Mean price: {}').format(self.plots_data[5])))
            self.text2.insert(tk.END, (('Corr. sales: {}').format(self.plots_data[6])))
        elif algorithm == 'Pure ANN' and plot_type == 'Benefits':
            self.plot1.plot(self.plots_data[0],self.plots_data[3])
            self.plot1.set_title('Benefits / Price (pure ANN algorithm)', size=12)
            self.plot1.set_ylabel('Benefits', fontsize = 12)
            self.text1.insert(tk.END, (('Optimal price: {}').format(self.plots_data[7])))
            self.text2.insert(tk.END, (('Corr. benefits: {}').format(self.plots_data[8])))
        elif algorithm == 'Pure ANN' and plot_type == 'Sales':
            self.plot1.plot(self.plots_data[0],self.plots_data[1])
            self.plot1.set_title('Sales / Price (pure ANN algorithm)', size=12)
            self.plot1.set_ylabel('Sales', fontsize = 12)
            self.text1.insert(tk.END, (('Mean price: {}').format(self.plots_data[5])))
            self.text2.insert(tk.END, (('Corr. sales: {}').format(self.plots_data[6])))
        
        self.text1.config(state=DISABLED)
        self.text2.config(state=DISABLED)
        self.plot1.set_xlabel('Price', fontsize = 12)
        self.canvas.draw()


def changeRightPlot(self):
    algorithm = self.algorithm2.get()
    plot_type = self.plot_type2.get()
    
    if self.plots_data != '':
        self.plot2.clear()
        self.text3.config(state='normal')
        self.text3.delete('1.0', tk.END)
        self.text4.config(state='normal')
        self.text4.delete('1.0', tk.END)
        if algorithm == 'Optimized' and plot_type == 'Benefits':
            self.plot2.plot(self.plots_data[0],self.plots_data[4])
            self.plot2.set_title('Benefits / Price (optimized algorithm)', size=12)
            self.plot2.set_ylabel('Benefits', fontsize = 12)
            self.text3.insert(tk.END, (('Optimal price: {}').format(self.plots_data[9])))
            self.text4.insert(tk.END, (('Corr. benefits: {}').format(self.plots_data[10])))
        elif algorithm == 'Optimized' and plot_type == 'Sales':
            self.plot2.plot(self.plots_data[0],self.plots_data[2])
            self.plot2.set_title('Sales / Price (optimized algorithm)', size=12)
            self.plot2.set_ylabel('Sales', fontsize = 12)
            self.text3.insert(tk.END, (('Mean price: {}').format(self.plots_data[5])))
            self.text4.insert(tk.END, (('Corr. sales: {}').format(self.plots_data[6])))
        elif algorithm == 'Pure ANN' and plot_type == 'Benefits':
            self.plot2.plot(self.plots_data[0],self.plots_data[3])
            self.plot2.set_title('Benefits / Price (pure ANN algorithm)', size=12)
            self.plot2.set_ylabel('Benefits', fontsize = 12)
            self.text3.insert(tk.END, (('Optimal price: {}').format(self.plots_data[7])))
            self.text4.insert(tk.END, (('Corr. benefits: {}').format(self.plots_data[8])))
        elif algorithm == 'Pure ANN' and plot_type == 'Sales':
            self.plot2.plot(self.plots_data[0],self.plots_data[1])
            self.plot2.set_title('Sales / Price (pure ANN algorithm)', size=12)
            self.plot2.set_ylabel('Sales', fontsize = 12)
            self.text3.insert(tk.END, (('Mean price: {}').format(self.plots_data[5])))
            self.text4.insert(tk.END, (('Corr. sales: {}').format(self.plots_data[6])))
        
        self.text3.config(state=DISABLED)
        self.text4.config(state=DISABLED)
        self.plot2.set_xlabel('Price', fontsize = 12)
        self.canvas2.draw()

application = Product()
application.main()