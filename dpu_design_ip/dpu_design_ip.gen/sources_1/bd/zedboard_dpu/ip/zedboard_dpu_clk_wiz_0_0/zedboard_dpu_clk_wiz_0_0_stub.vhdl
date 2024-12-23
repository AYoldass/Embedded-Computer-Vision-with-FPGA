-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2020.1 (win64) Build 2902540 Wed May 27 19:54:49 MDT 2020
-- Date        : Sun Dec 22 15:35:50 2024
-- Host        : DESKTOP-MU42VMQ running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode synth_stub
--               c:/Users/yasin/Desktop/dpu_design_ip/dpu_design_ip.srcs/sources_1/bd/zedboard_dpu/ip/zedboard_dpu_clk_wiz_0_0/zedboard_dpu_clk_wiz_0_0_stub.vhdl
-- Design      : zedboard_dpu_clk_wiz_0_0
-- Purpose     : Stub declaration of top-level module interface
-- Device      : xc7z020clg484-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity zedboard_dpu_clk_wiz_0_0 is
  Port ( 
    clk_out150M : out STD_LOGIC;
    clk_out300M : out STD_LOGIC;
    resetn : in STD_LOGIC;
    locked : out STD_LOGIC;
    clk_in1 : in STD_LOGIC
  );

end zedboard_dpu_clk_wiz_0_0;

architecture stub of zedboard_dpu_clk_wiz_0_0 is
attribute syn_black_box : boolean;
attribute black_box_pad_pin : string;
attribute syn_black_box of stub : architecture is true;
attribute black_box_pad_pin of stub : architecture is "clk_out150M,clk_out300M,resetn,locked,clk_in1";
begin
end;
