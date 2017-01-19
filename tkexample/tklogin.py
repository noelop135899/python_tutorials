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
    user_file.close
        
def user_signup():
    
    def sign_up_new_user():
        nun = new_usr_name.get()
        np = new_pwd.get()
        npc = new_pwd_confirm.get()
        with open('user_info_file.pickle','rb') as user_file:
            file_user_info = pickle.load(user_file)
        
        if nun in file_user_info:
            tk.messagebox.showerror(message='this user name is registered')
        elif np != npc:
            tk.messagebox.showerror(message='the password have be the same')
        else:
            file_user_info[nun] = np
            with open('user_info_file.pickle','wb') as user_file:
                pickle.dump(file_user_info,user_file)

            tk.messagebox.showerror(message='New User sign up is complete , Welcome ')
            window_sign_up.destroy()
        user_file.close
        
    def clear_entry_text():
        new_usr_name.set('')
        new_pwd.set('')
        new_pwd_confirm.set('')
    
    window_sign_up = tk.Toplevel(window)
    window_sign_up.geometry('350x200')
    window_sign_up.title('Welcome Sing up')

    new_usr_name = tk.StringVar()
    new_usr_name.set('example@python.com')
    tk.Label(window_sign_up,text='New user name :').place(x=10,y=10)
    nun = tk.Entry(window_sign_up,textvariable=new_usr_name).place(x=150,y=10)

    new_pwd = tk.StringVar()
    tk.Label(window_sign_up,text='Password :').place(x=10,y=50)
    npwd = tk.Entry(window_sign_up,textvariable=new_pwd,show='*').place(x=150,y=50)

    new_pwd_confirm = tk.StringVar()
    tk.Label(window_sign_up,text='Password Confirm :').place(x=10,y=90)
    npwdcm = tk.Entry(window_sign_up,textvariable=new_pwd_confirm,show='*').place(x=150,y=90)

    sunubtn = tk.Button(window_sign_up,text='new',command=sign_up_new_user).place(x=150,y=130)
    cetbtn = tk.Button(window_sign_up,text='clear',command=clear_entry_text).place(x=250,y=130)


btn_login = tk.Button(window, text='login', command=user_login).place(x=170,y=230)
btn_signup = tk.Button(window, text='sign up', command=user_signup).place(x=270,y=230)

window.mainloop()
