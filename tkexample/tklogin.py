import tkinter as tk
import pickle
import tkinter.messagebox

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
entry_password = tk.Entry(window,textvariable=var_password,show='*').place(x=160,y=200)

def user_login():
    
    usr_name = var_username.get()
    usr_pwd = var_password.get()
    try:
        with open('user_info_file.pickle','rb') as user_file:
            user_info = pickle.load(user_file)
    except FileNotFoundError:
        with open('user_info_file.pickle','wb') as user_file:
            user_info = {'admin': 'admin'}
            pickle.dump(user_info, user_file)
    if usr_name in user_info:
        if usr_pwd == user_info[usr_name]:
            tk.messagebox.showinfo(title='Welcome', message='How are you? ' + usr_name)
        else:
            tk.messagebox.showerror(message='Error, your password is wrong, try again.')
            var_password.set('')
    else:
        is_sign_up = tk.messagebox.askyesno(title='Welcome',
                               message='Your user name is not registered. Sign up today?')
        if is_sign_up:
            user_signup()
        else:
            pass
        
def user_signup():
    tk.messagebox.showinfo(message='user_signup')

btn_login = tk.Button(window, text='login', command=user_login).place(x=170,y=230)
btn_signup = tk.Button(window, text='sign up', command=user_signup).place(x=270,y=230)

window.mainloop()
