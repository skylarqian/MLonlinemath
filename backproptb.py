import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


@cocotb.test()
async def backprop_test(dut):
    
    dataset = {"currW0": , "currb0":, "currW1":, "currb1":, 
           "predicted state":, "real state":, "inputs": ,"softmax":, "reluout":}
    clock = Clock(dut.clk, 2, units="ns")
    cocotb.start_soon(clock.start())
    print("starting test")
    dut.