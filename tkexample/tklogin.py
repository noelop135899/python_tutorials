import Tkinter as tk

window = tk.Tk()
window.title('Welcome to Noelop Python')
window.geometry('450x300')

canvas = tk.Canvas(window, height=200, width=500)
image_file = tk.PhotoImage(file='welcome.gif')
image = canvas.create_image(0,0, anchor='nw',image=image_file)
canvas.pack()

tk.Label(window,text='User name :').place(x=50,y=150)
tk.Label(window,text='Password :').place(x=50,y=200)

var_username = tk.StringVar()
entry_username = tk.Entry(window,textvariable=var_username).place(x=160,y=150)
var_username.set('example@python.com')
var_password = tk.StringVar()
entry_password = tk.Entry(window,textvariable=var_password).place(x=160,y=200)

btn_login = tk.Button(window, text='login', command='user_login').place(x=170,y=230)
btn_signup = tk.Button(window, text='sign up', command='user_signup').place(x=270,y=230)

def user_login():
    pass
def user_signup():
    pass
window.mainloop()
