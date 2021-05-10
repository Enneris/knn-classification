#Импорт библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import seaborn as sns

df=pd.read_csv('income_evaluation.csv')

#убирает \n,\t
for x in df.columns:
    x_new=x.strip()
    df=df.rename(columns={x:x_new})

##################
# Убираем лишние колонки
##################

data = df.drop(['fnlwgt', 'capital-gain', 'capital-loss', 'education-num', ], axis=1)

for column in data[['workclass','education','marital-status','occupation','race', 'sex']]:
    data[column]=data[column].str.strip()

#Группирует страны по регионам
data=data.replace([' United-States',' Canada', ' Cuba',' Jamaica',' Dominican-Republic',' El-Salvador',' Guatemala',' Haiti',' Honduras',' Mexico',' Nicaragua',' Outlying-US(Guam-USVI-etc)',' Puerto-Rico',
                   ' Trinadad&Tobago'],'North America')
data=data.replace([' India',' Cambodia',' China',' Hong',' Iran',' Japan',' Laos',' Philippines',' Taiwan',' Thailand',' Vietnam'],'Asia')
data=data.replace([' Ecuador',' Columbia',' Peru'],'South America')
data=data.replace([' England',' France',' Germany',' Greece',' Holand-Netherlands',' Hungary',' Ireland',' Italy',' Poland',' Portugal',' Scotland',' Yugoslavia'],'Europe')
data=data.replace([' South',' ?'],'Other')

#Таблица на русском для графиков
data_plots=data.copy()
data_plots=data_plots.rename(columns={'age':'Возраст','workclass':'Работа','education':'Образование',
                                      'marital-status':'Семейное положение','occupation':'Профессия','relationship':'Отношения',
                                      'race':'Раса','sex':'Пол','hours-per-week':'Рабочие часы','native-country':'Регион',
                                      'income':'Доход'})

##################
# Меняем качественные переменные на числа
##################

wcl_titles=data['workclass'].unique()
educ_titles=data['education'].unique()
native_titles=data['native-country'].unique()
mst_titles=data['marital-status'].unique()
occ_titles=data['occupation'].unique()
rel_titles=data['relationship'].unique()
race_titles=data['race'].unique()
hpw_titles=data['hours-per-week'].unique()
sex_titles=data['sex'].unique()

# Работа
lb_workclass = preprocessing.LabelEncoder()
lb_workclass.fit(['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', '?','Self-emp-inc', 'Without-pay',
                  'Never-worked'])
data.iloc[:,1]=lb_workclass.transform(data.iloc[:,1])
wcl_list=['Правительство субъекта','Незарегистрированный предприниматель','Частная компания', 'Федеральное правительство',
          'Местное правительство','Другое','Зарегистрированный предприниматель','Без оплаты','Безработный']
wcl_dict={x:y for x,y in zip(wcl_titles,wcl_list)}

data_plots=data_plots.replace(['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', '?', 'Self-emp-inc',
                               'Without-pay', 'Never-worked'],wcl_list)

# Регион
lb_native = preprocessing.LabelEncoder()
lb_native.fit(['North America', 'Asia', 'Other', 'Europe', 'South America'])
data.iloc[:,9]=lb_native.transform(data.iloc[:,9])
native_list=['Северная Америка','Азия','Другое', 'Европа','Южная Америка']
native_dict={x:y for x,y in zip(native_titles,native_list)}

data_plots=data_plots.replace(['North America', 'Asia', 'Other', 'Europe', 'South America'],native_list)

# Образование
lb_educ = preprocessing.LabelEncoder()
lb_educ.fit(['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc', '7th-8th',
             'Doctorate', 'Prof-school', '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
data.iloc[:,2]=lb_educ.transform(data.iloc[:,2])
educ_list=['Бакалавр','Среднее образование','11 классов','Магистр','9 классов','Незаконченное высшее','Другое','СПО',
           '7-8 классов','Докторская','Профессиональная школа','5-6 классов','10 классов','1-4 класс','Дошкольное','12 классов']
educ_dict={x:y for x,y in zip(educ_titles,educ_list)}

data_plots=data_plots.replace(['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc',
                               '7th-8th', 'Doctorate', 'Prof-school', '5th-6th', '10th','1st-4th', 'Preschool', '12th'],
                              educ_list)

# Семейное положение
lb_marital = preprocessing.LabelEncoder()
lb_marital.fit(['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
                'Widowed'])
data.iloc[:,3]=lb_marital.transform(data.iloc[:,3])
mst_list=['Никогда не был(а) в браке','В браке с гражданским','Разведен(а)','В браке, раздельно от супруга',
          'В браке, раздельно от супруга (по правовой причине)','В браке с военноослужащим','Овдовевший(ая)']
mst_dict={x:y for x,y in zip(mst_titles,mst_list)}

data_plots=data_plots.replace(['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated',
                               'Married-AF-spouse', 'Widowed'], mst_list)

# Профессия
lb_occup = preprocessing.LabelEncoder()
lb_occup.fit(['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales',
              'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', '?',
              'Protective-serv', 'Armed-Forces', 'Priv-house-serv'])
data.iloc[:,4]=lb_occup.transform(data.iloc[:,4])
occ_list=['Администратор/секретарь','Руководитель','Уборщик','Профессиональная специальность','Услуги','Продажи',
          'Ремесло/ремонт','Перевозки','Сельское хозяйство/рыбная ловля','Оператор машины/станка','Техническая поддержка',
          'Другое','Служба защиты','Военные силы','Служба жилья']
occ_dict={x:y for x,y in zip(occ_titles,occ_list)}

data_plots=data_plots.replace(['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service',
                               'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct',
                                'Tech-support', '?', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'], occ_list)

# Отношение
lb_rel = preprocessing.LabelEncoder()
lb_rel.fit([' Not-in-family', ' Husband', ' Wife', ' Own-child', ' Unmarried', ' Other-relative'])
data.iloc[:,5]=lb_rel.transform(data.iloc[:,5])
rel_list=['Без семьи','Муж','Жена','Есть ребенок','Не в браке','Другое']
rel_dict={x:y for x,y in zip(rel_titles,rel_list)}

data_plots=data_plots.replace([' Not-in-family', ' Husband', ' Wife', ' Own-child', ' Unmarried', ' Other-relative'], rel_list)

# Раса
lb_race = preprocessing.LabelEncoder()
lb_race.fit(['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
data.iloc[:,6]=lb_race.transform(data.iloc[:,6])
race_list=['Белый(ая)','Черный(ая)','Азиат/тихоокеанского происхождения','Американские индейцы/эскимосы','Другое']
race_dict={x:y for x,y in zip(rel_titles,race_list)}

data_plots=data_plots.replace(['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], race_list)

# Пол
lb_sex = preprocessing.LabelEncoder()
lb_sex.fit(['Male','Female'])
data.iloc[:,7]=lb_sex.transform(data.iloc[:,7])
sex_list=['Мужской','Женский']
sex_dict={x:y for x,y in zip(sex_titles,sex_list)}

data_plots=data_plots.replace(['Male','Female'], sex_list)

#Группирует числовые значения для графиков
data_plots_count=data_plots.copy()

data_plots_count = data_plots_count.replace(list(range(1,11)),'1-10')
data_plots_count = data_plots_count.replace(list(range(11,21)),'11-20')
data_plots_count = data_plots_count.replace(list(range(21,31)),'21-30')
data_plots_count = data_plots_count.replace(list(range(31,41)),'31-40')
data_plots_count = data_plots_count.replace(list(range(41,51)),'41-50')
data_plots_count = data_plots_count.replace(list(range(51,61)),'51-60')
data_plots_count = data_plots_count.replace(list(range(61,71)),'61-70')
data_plots_count = data_plots_count.replace(list(range(71,81)),'71-80')
data_plots_count = data_plots_count.replace(list(range(81,91)),'81-90')
data_plots_count = data_plots_count.replace(list(range(91,100)),'91-99')

X=data.iloc[:,:-1]
y=data[['income']]

##################
# Тренировочные/тестовые данные
##################

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=4)

##################
# GUI
##################

income_values = ['<=50K', '>50K']

##################
# Основное окно
##################
root = Tk()
root.geometry("700x800")
root.title('Анализ данных')
root.iconbitmap('small_icon.ico')
root.configure(background="white")

# Лого
logo = Image.open('icon.png')
new_logo = Image.new("RGBA", logo.size, "WHITE")
new_logo.paste(logo, (0, 0), logo)
new_logo = ImageTk.PhotoImage(new_logo)
logo_label = tk.Label(image=new_logo, bg="white")
logo_label.image = new_logo
logo_label.place(x=150, y=-40)

# Подписи
Label(root, text="Возраст", font=('Candara', 13, 'bold'), bg="white").place(x=180, y=135)
Label(root, text="Категория работы", font=('Candara', 13, 'bold'), bg="white").place(x=100, y=180)
Label(root, text="Образование", font=('Candara', 13, 'bold'), bg="white").place(x=135, y=225)
Label(root, text="Семейное положение", font=('Candara', 13, 'bold'), bg="white").place(x=80, y=270)
Label(root, text="Профессия", font=('Candara', 13, 'bold'), bg="white").place(x=150, y=315)
Label(root, text="Отношение к семье", font=('Candara', 13, 'bold'), bg="white").place(x=90, y=360)
Label(root, text="Раса", font=('Candara', 13, 'bold'), bg="white").place(x=200, y=405)
Label(root, text="Пол", font=('Candara', 13, 'bold'), bg="white").place(x=200, y=450)
Label(root, text="Родной регион", font=('Candara', 13, 'bold'), bg="white").place(x=120, y=495)
Label(root, text="Рабочие часы в неделю", font=('Candara', 13, 'bold'), bg="white").place(x=50, y=540)
Label(root, text="Результат: ", font=('Candara', 14, 'bold'), bg="white", relief='solid', width=10).place(x=50, y=670)

# Переменные
age = IntVar()
wcl = StringVar()
wcl.set(wcl_list[0])
educ = StringVar()
educ.set(educ_list[0])
mst = StringVar()
mst.set(mst_list[0])
occ = StringVar()
occ.set(occ_list[0])
rel = StringVar()
rel.set(rel_list[0])
race = StringVar()
race.set(race_list[0])
sex = StringVar()
sex.set(sex_list[0])
native = StringVar()
native.set(native_list[0])
hpw = IntVar()

c_age_var = IntVar()
c_wcl_var = IntVar()
c_educ_var = IntVar()
c_mst_var = IntVar()
c_occ_var = IntVar()
c_rel_var = IntVar()
c_race_var = IntVar()
c_sex_var = IntVar()
c_native_var = IntVar()
c_hpw_var = IntVar()


# Проверка энтри
def validate(P):
    if len(P) == 0:
        return True
    elif 0 <= len(P) <= 2 and P.isdigit():
        return True
    else:
        return False


vcmd = (root.register(validate), '%P')

# Виджеты
Entry(root, text=age, width=10, validate="key", validatecommand=vcmd).place(x=250, y=135)
om_wcl = OptionMenu(root, wcl, *wcl_list)
om_wcl.config(bg="#cfebf0", font=('Candara', 12))
om_wcl.place(x=250, y=180)
om_educ = OptionMenu(root, educ, *educ_list)
om_educ.config(bg="#cfebf0", font=('Candara', 12))
om_educ.place(x=250, y=225)
om_mst = OptionMenu(root, mst, *mst_list)
om_mst.config(bg="#cfebf0", font=('Candara', 12))
om_mst.place(x=250, y=270)
om_occ = OptionMenu(root, occ, *occ_list)
om_occ.config(bg="#cfebf0", font=('Candara', 12))
om_occ.place(x=250, y=315)
om_rel = OptionMenu(root, rel, *rel_list)
om_rel.config(bg="#cfebf0", font=('Candara', 12))
om_rel.place(x=250, y=360)
om_race = OptionMenu(root, race, *race_list)
om_race.config(bg="#cfebf0", font=('Candara', 12))
om_race.place(x=250, y=405)
om_sex = OptionMenu(root, sex, *sex_list)
om_sex.config(bg="#cfebf0", font=('Candara', 12))
om_sex.place(x=250, y=450)
om_nat = OptionMenu(root, native, *native_list)
om_nat.config(bg="#cfebf0", font=('Candara', 12))
om_nat.place(x=250, y=495)
Entry(root, text=hpw, width=10, validate="key", validatecommand=vcmd).place(x=250, y=540)


##################
# Функции
##################

# Проверка пустых значений
def check_number(var):
    if var == '':
        messagebox.showerror("Ошибка", "Необходимо заполнить все поля")
        return False
    else:
        return True


# Круговая диаграмма
def make_plot_pie(figure, frame, column_name, input_get, title):
    ax = figure.add_subplot(111)
    bar = FigureCanvasTkAgg(figure, frame)
    colors = ['#65c72e', '#3bb8d2']
    bar.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH)
    new_data = data_plots[data_plots[column_name] == input_get]
    new_data = pd.Series(new_data['Доход']).value_counts()
    ax.pie(new_data, labels=income_values, explode=[0, 0], autopct='%1.1f%%', radius=1, shadow=True, colors=colors)
    ax.set_title(title, fontsize=9)


# гистограмма по категориям
def make_countplot(column_name):
    f, ax = plt.subplots(figsize=(7,6))
    sns.countplot(x='Доход', hue=column_name, data=data_plots_count, palette='Set2').set(ylabel='Количество людей')
    return f


# Проверка наличия данных с определенным параметром
def have_data(column_name, input_get):
    new_data = data_plots[data_plots[column_name] == input_get]
    if pd.Series(new_data['Доход']).value_counts().empty:
        return False
    else:
        return True


# Для чекбоксов
def toggle(var):
    var.set(not var.get())


# Прокрутка окна
def make_scrollbar(window, pad):
    main_frame = Frame(window)
    main_frame.pack(fill=BOTH, expand=1)

    my_canvas = Canvas(main_frame, height=600)
    my_canvas.pack(side=LEFT, fill=BOTH, expand=1, padx=pad)

    my_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
    my_scrollbar.pack(side=RIGHT, fill=Y)

    # Прокрутка скроллбара мышкой
    def _on_mouse_wheel(event):
        my_canvas.yview_scroll(-1 * int((event.delta / 120)), "units")

    my_canvas.configure(yscrollcommand=my_scrollbar.set)
    my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion=my_canvas.bbox("all")))
    my_canvas.bind_all("<MouseWheel>", _on_mouse_wheel)

    second_frame = Frame(my_canvas, width=700, height=600)
    second_frame.configure(background='white')

    my_canvas.create_window((0, 0), window=second_frame, anchor='nw')
    my_canvas.configure(background='white')
    main_frame.configure(background='white')

    return second_frame


# делает словарь и список для графиков
def collect_data_for_charts(age, wcl, educ, mst, occ, rel, race, sex, hpw, native, c_age_var, c_wcl_var, c_educ_var,
                            c_mst_var, c_occ_var,
                            c_rel_var, c_race_var, c_sex_var, c_native_var, c_hpw_var):
    chart_list_a = [['C возрастом ' + str(age.get()) + ' лет', age.get()], ['В категории:' + wcl.get(), wcl.get()],
                    ['С образованием: ' + educ.get(), educ.get()], ['С семейным положением:' + mst.get(), mst.get()],
                    ['В професии: ' + occ.get(), occ.get()], ['Относящихся к семье как: ' + rel.get(), rel.get()],
                    ['С расой:' + race.get(), race.get()], ['С полом: ' + sex.get(), sex.get()],
                    ['Работающих ' + str(hpw.get()) + ' часов в неделю:', hpw.get()],
                    ['Из региона: ' + native.get(), native.get()]]

    chart_dict = {x: y for x, y in zip(data_plots.columns.values, chart_list_a)}
    charts_to_make = []
    charts_list = [c_age_var.get(), c_wcl_var.get(), c_educ_var.get(), c_mst_var.get(), c_occ_var.get(),
                   c_rel_var.get(),
                   c_race_var.get(), c_sex_var.get(), c_native_var.get(), c_hpw_var.get()]

    for ind, elem in enumerate(charts_list):
        if elem == 1:
            charts_to_make.append(data_plots.columns.values[ind])

    return charts_to_make, chart_dict


# Индекс элемента
def get_index(arr, val):
    return arr.index(val.get())


##################
# Окно с выбором графика
##################
def info():
    n = Tk()
    n.geometry("550x350")
    n.title('Анализ данных')
    n.iconbitmap('small_icon.ico')
    n.configure(background='white')

    Button(n, text='Выход', width=9, height=1, command=n.destroy, font=('Candara', 13, 'bold'), bg='#cf6c70',
           fg='black').place(x=450, y=0)

    c_age = Checkbutton(n, text="Возраст", font=('Candara', 12), bg='white', command=lambda: toggle(c_age_var)).place(
        x=250,
        y=40)
    c_wcl = Checkbutton(n, text="Категория работы", font=('Candara', 12), bg='white',
                        command=lambda: toggle(c_wcl_var)).place(x=250, y=65)
    c_educ = Checkbutton(n, text="Образование", font=('Candara', 12), bg='white',
                         command=lambda: toggle(c_educ_var)).place(x=250, y=90)
    c_mst = Checkbutton(n, text="Семейное положение", font=('Candara', 12), bg='white',
                        command=lambda: toggle(c_mst_var)).place(x=250, y=115)
    c_occ = Checkbutton(n, text="Профессия", font=('Candara', 12), bg='white', command=lambda: toggle(c_occ_var)).place(
        x=250,
        y=140)
    c_rel = Checkbutton(n, text="Отношение к семье", font=('Candara', 12), bg='white',
                        command=lambda: toggle(c_rel_var)).place(x=250, y=165)
    c_race = Checkbutton(n, text="Раса", font=('Candara', 12), bg='white', command=lambda: toggle(c_race_var)).place(
        x=250,
        y=190)
    c_sex = Checkbutton(n, text="Пол", font=('Candara', 12), bg='white', command=lambda: toggle(c_sex_var)).place(x=250,
                                                                                                                  y=215)
    c_native = Checkbutton(n, text="Родной регион", font=('Candara', 12), bg='white',
                           command=lambda: toggle(c_native_var)).place(x=250, y=240)
    c_hpw = Checkbutton(n, text="Рабочие часы в неделю", font=('Candara', 12), bg='white',
                        command=lambda: toggle(c_hpw_var)).place(x=250, y=265)

    Button(n, text='Круговая диаграмма', width=18, height=2, command=pie, font=('Candara', 13, 'bold'), bg='#9fd7e0',
           fg='black').place(x=45, y=40)

    Button(n, text='Гистограмма', width=16, height=2, command=countplot, font=('Candara', 13, 'bold'), bg='#9fd7e0',
           fg='black').place(x=55, y=110)

    n.resizable(0, 0)
    n.mainloop()


##################
# Окно с круговой диаграммой
##################
def pie():
    try:
        pie_window = Toplevel()
        pie_window.geometry("600x700")
        pie_window.title('Анализ данных')
        pie_window.iconbitmap('small_icon.ico')
        pie_window.configure(background='white')

        second_frame = make_scrollbar(pie_window, 60)
        inc_share_label = tk.Label(second_frame, text='Распределение дохода у людей:', font=('Candara', 18, 'bold'),
                                   bg='white').pack()


        charts_to_make, chart_dict = collect_data_for_charts(age, wcl, educ, mst, occ, rel, race, sex, hpw, native,
                                                             c_age_var, c_wcl_var,
                                                             c_educ_var, c_mst_var, c_occ_var, c_rel_var, c_race_var,
                                                             c_sex_var,
                                                             c_native_var, c_hpw_var)

        for elem in charts_to_make:
            if have_data(elem, chart_dict[elem][1]):
                figure = plt.Figure(figsize=(4, 3), dpi=100)
                make_plot_pie(figure, second_frame, elem, chart_dict[elem][1], chart_dict[elem][0])

        pie_window.resizable(0, 0)
        pie_window.mainloop()
    except TclError:
        messagebox.showerror("Ошибка", "Необходимо заполнить все поля")
        pie_window.destroy()


##################
# Окно с гистограммой
##################
def countplot():
    try:
        count_window = Toplevel()
        count_window.geometry("700x650")
        count_window.title('Анализ данных')
        count_window.iconbitmap('small_icon.ico')
        count_window.configure(background='white')

        second_frame = make_scrollbar(count_window, 20)

        inc_share_label = tk.Label(second_frame, text='Распределение дохода у людей', font=('Candara', 18, 'bold'),
                                   bg='white').pack()

        charts_to_make = collect_data_for_charts(age, wcl, educ, mst, occ, rel, race, sex, hpw, native, c_age_var,
                                                 c_wcl_var, c_educ_var,
                                                 c_mst_var, c_occ_var, c_rel_var, c_race_var, c_sex_var, c_native_var,
                                                 c_hpw_var)

        for elem in charts_to_make[0]:
            fig = make_countplot(elem)
            canvas = FigureCanvasTkAgg(fig, master=second_frame)
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH)

        count_window.resizable(0, 0)
        count_window.mainloop()
    except TclError:
        messagebox.showerror("Ошибка", "Необходимо заполнить все поля")
        count_window.destroy()


##################
# КНН
##################
def model():
    try:
        # Ввод данных
        wcl_value = get_index(wcl_list, wcl)
        educ_value = get_index(educ_list, educ)
        mst_value = get_index(mst_list, mst)
        occ_value = get_index(occ_list, occ)
        rel_value = get_index(rel_list, rel)
        race_value = get_index(race_list, race)
        sex_value = get_index(sex_list, sex)
        native_value = get_index(native_list, native)

        # Классифаер
        neigh = KNeighborsClassifier(n_neighbors=9).fit(X_train, np.ravel(y_train))
        x_test = [[age.get(), wcl_value, educ_value, mst_value, occ_value, rel_value, race_value, sex_value, hpw.get(),
                   native_value]]

        y_pred = neigh.predict(x_test)

        # Вывод рез-та
        if y_pred[0] == ' <=50K':
            result_text = 'Меньше 50 тысяч в год'
        else:
            result_text = 'Больше 50 тысяч в год'

        proba = neigh.predict_proba(x_test).max() * 100
        Label(root, text=result_text, font=('Candara', 15, 'bold'), bg="white", width=20).place(x=159, y=670)
        Label(root, text='С вероятностью в ' + str(round(proba, 2)) + '%', font=('Candara', 13), bg="white",
              width=20).place(x=171, y=700)

    except TclError:
        messagebox.showerror("Ошибка", "Необходимо заполнить все поля")


# Кнопочки
Button(root, text='Результат', width=15, height=2, command=model, font=('Candara', 13, 'bold'), bg='#9fd7e0',
       fg='black').place(x=50,
                         y=585)

Button(root, text='Узнать больше', width=15, height=2, command=info, font=('Candara', 13, 'bold'), bg='#9fd7e0',
       fg='black').place(x=250, y=585)

Button(root, text='Выход', width=10, height=1, command=root.destroy, font=('Candara', 13, 'bold'), bg='#cf6c70',
       fg='black').place(x=550, y=725)

root.resizable(0, 0)
root.mainloop()




