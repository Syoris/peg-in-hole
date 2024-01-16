"""
To list all of Kinova Error codes.

See CommandLayer.h and CommunicationLayer.h for their values.
"""

from enum import Enum


class ErrorCode(Enum):
    # No error, everything is fine.
    NO_ERROR_KINOVA = 1

    # We know that an error has occured but we don't know where it comes from.
    UNKNOWN_ERROR = 666

    ERROR_INIT_API = 2001  # Error while initializing the API
    ERROR_LOAD_COMM_DLL = 2002  # Error while loading the communication layer

    # Those 3 codes are mostly for internal use
    JACO_NACK_FIRST = 2003
    JACO_COMM_FAILED = 2004
    JACO_NACK_NORMAL = 2005

    # Unable to initialize the communication layer.
    ERROR_INIT_COMM_METHOD = 2006

    # Unable to load the Close() function from the communication layer.
    ERROR_CLOSE_METHOD = 2007

    # Unable to load the GetDeviceCount() function from the communication layer.
    ERROR_GET_DEVICE_COUNT_METHOD = 2008

    # Unable to load the SendPacket() function from the communication layer.
    ERROR_SEND_PACKET_METHOD = 2009

    # Unable to load the SetActiveDevice() function from the communication layer.
    ERROR_SET_ACTIVE_DEVICE_METHOD = 2010

    # Unable to load the GetDeviceList() function from the communication layer.
    ERROR_GET_DEVICES_LIST_METHOD = 2011

    # Unable to initialized the system semaphore.
    ERROR_SEMAPHORE_FAILED = 2012

    # Unable to load the ScanForNewDevice() function from the communication layer.
    ERROR_SCAN_FOR_NEW_DEVICE = 2013

    # Unable to load the GetActiveDevice function from the communication layer.
    ERROR_GET_ACTIVE_DEVICE_METHOD = 2014

    # Unable to load the OpenRS485_Activate() function from the communication layer.
    ERROR_OPEN_RS485_ACTIVATE = 2015

    # A function's parameter is not valid.
    ERROR_INVALID_PARAM = 2100

    # The API is not initialized.
    ERROR_API_NOT_INITIALIZED = 2101

    # Unable to load the InitDataStructure() function from the communication layer.
    ERROR_INIT_DATA_STRUCTURES_METHOD = 2102

    # Unable to load the USB library.
    ERROR_LOAD_USB_LIBRARY = 1001

    # Unable to access the Open method from the USB library.
    ERROR_OPEN_METHOD = 1002

    # Unable to access the Write method from the USB library.
    ERROR_WRITE_METHOD = 1003

    # Unable to access the Read method from the USB library.
    ERROR_READ_METHOD = 1004

    # Unable to access the Read Int method from the USB library.
    ERROR_READ_INT_METHOD = 1005

    # Unable to access the Free Library method from the USB library.
    ERROR_FREE_LIBRARY = 1006

    # There is a problem with the USB connection between the device and the computer.
    ERROR_JACO_CONNECTION = 1007

    # Unable to claim the USB interface.
    ERROR_CLAIM_INTERFACE = 1008

    # Unknown type of device.
    ERROR_UNKNOWN_DEVICE = 1009

    # The functionality you are trying to use has not been initialized.
    ERROR_NOT_INITIALIZED = 1010

    # The USB library cannot find the device.
    ERROR_LIBUSB_NO_DEVICE = 1011

    # The USB Library is bussy and could not perform the action.
    ERROR_LIBUSB_BUSY = 1012

    # The functionality you are trying to perform is not supported by the version installed.
    ERROR_LIBUSB_NOT_SUPPORTED = 1013

    # Unknown error while sending a packet.
    ERROR_SENDPACKET_UNKNOWN = 1014

    # Cannot find the requested device.
    ERROR_NO_DEVICE_FOUND = 1015

    # The operation was not entirely completed :)
    ERROR_OPERATION_INCOMPLETED = 1016

    # Handle used is not valid.
    ERROR_RS485_INVALID_HANDLE = 1017

    # An overlapped I/O operation is in progress but has not completed.
    ERROR_RS485_IO_PENDING = 1018

    # Not enough memory to complete the opreation.
    ERROR_RS485_NOT_ENOUGH_MEMORY = 1019

    # The operation has timed out.
    ERROR_RS485_TIMEOUT = 1020

    # You are trying to call a USB function but the OpenRS485_Activate has been called. Functions are no longer available
    ERROR_FUNCTION_NOT_ACCESSIBLE = 1021

    # No response timeout reached
    ERROR_COMM_TIMEOUT = 1022

    # If the robot answered a NACK to our command
    ERROR_NACK_RECEIVED = 9999
