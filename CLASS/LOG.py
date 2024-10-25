
def make_print_to_file(path='./',content = None):
    '''
    A function to redirect print statements to a log file.

    :param path: The path to the directory where the log file should be saved.
    :return: None
    '''
    import sys
    import os
    import datetime
 
    class Logger(object):
        def __init__(self, filename="Default.log", path="./log"):

            self.terminal = sys.stdout # terminal是标准输出，即print函数输出的位置
            os.makedirs(path, exist_ok=True)
            self.path= os.path.join(path, filename) # path是文件保存的路径
            
            self.log_file = open(self.path, "a", encoding='utf8',) # log_file是文件对象，用于写入文件
            
            print("Saving logs to:", os.path.join(self.path, filename)) # 打印日志保存的路径
 
        def write(self, message):
            '''
            Writes the message to both the terminal and the log file.
            :param message: The message to be written.
            '''
            self.terminal.write(message) # 将message写入到terminal，即标准输出
            self.log_file.write(message) # 将message写入到log_file，即文件
            self.log_file.flush() # 刷新缓存区，即将缓存区的内容写入到文件(这个地方一定要刷新，不然的话，文件中会没有内容)
 
        def flush(self):
            pass
 
    # Create a new log file with the current date as the filename
    fileName = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
    sys.stdout = Logger(fileName + '.log', path=path)
 
    # Print a header to indicate that all subsequent print statements will be logged
    print("Logging started for:", fileName.center(60,'*'))
    default_information = "The log does not have any other circumstances to explain.\n"
    if content is None:
        print("Logging for content:", default_information)
    else:
        print(f"Logging for content:{content}\n")

    # Return the logger object
    return sys.stdout.log_file




if __name__ == '__main__':
    content = f"""
    参数如下：topk={topk}，chunk_size={chunk_size}, chunk_overlap={chunk_overlap}
    """
    # 加载日志
    log_file = make_print_to_file(path='./log',content = content)

    '''自己的代码部分'''

    log_file.close()