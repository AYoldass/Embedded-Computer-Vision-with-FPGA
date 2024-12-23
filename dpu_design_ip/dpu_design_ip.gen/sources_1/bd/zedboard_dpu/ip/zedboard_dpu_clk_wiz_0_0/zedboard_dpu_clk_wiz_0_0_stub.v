// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2020.1 (win64) Build 2902540 Wed May 27 19:54:49 MDT 2020
// Date        : Sun Dec 22 15:35:50 2024
// Host        : DESKTOP-MU42VMQ running 64-bit major release  (build 9200)
// Command     : write_verilog -force -mode synth_stub
//               c:/Users/yasin/Desktop/dpu_design_ip/dpu_design_ip.srcs/sources_1/bd/zedboard_dpu/ip/zedboard_dpu_clk_wiz_0_0/zedboard_dpu_clk_wiz_0_0_stub.v
// Design      : zedboard_dpu_clk_wiz_0_0
// Purpose     : Stub declaration of top-level module interface
// Device      : xc7z020clg484-1
// --------------------------------------------------------------------------------

// This empty module with port declaration file causes synthesis tools to infer a black box for IP.
// The synthesis directives are for Synopsys Synplify support to prevent IO buffer insertion.
// Please paste the declaration into a Verilog source file or add the file as an additional source.
module zedboard_dpu_clk_wiz_0_0(clk_out150M, clk_out300M, resetn, locked, 
  clk_in1)
/* synthesis syn_black_box black_box_pad_pin="clk_out150M,clk_out300M,resetn,locked,clk_in1" */;
  output clk_out150M;
  output clk_out300M;
  input resetn;
  output locked;
  input clk_in1;
endmodule
