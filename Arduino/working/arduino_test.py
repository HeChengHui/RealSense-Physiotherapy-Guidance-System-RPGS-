import serial
ser = serial.Serial('COM5', 9600)
ser.xonxoff=1
ser.flushInput()

# while True:
#     try:
#         ser_bytes = ser.readline()
#         decoded_bytes = float(ser_bytes[0:len(ser_bytes)-2].decode("utf-8"))
#         print(decoded_bytes)
#     except:
#         print("Keyboard Interrupt")
#         break

while True:
    ser_bytes = ser.readline()
    decoded_bytes = ser_bytes[0:len(ser_bytes)].decode("utf-8").strip('\n').split(',')
    print(decoded_bytes)
    # print(type(decoded_bytes))
    # list_values = decoded_bytes.strip('\n').split(',')
    # prin(list_values)
    # if"\x00" not in decoded_bytes[0]:
    #     float_list_values = [float(i) for i in decoded_bytes]
    #     print(float_list_values)
    # highest_list_values = max(decoded_bytes)
    if"\x00" not in decoded_bytes[0]:
        float_list_values = [float(i) for i in decoded_bytes]
        highest_list_values = max(float_list_values, key=abs)
        highest_axis_index = float_list_values.index(highest_list_values)
        print(highest_list_values)
        print(highest_axis_index)
