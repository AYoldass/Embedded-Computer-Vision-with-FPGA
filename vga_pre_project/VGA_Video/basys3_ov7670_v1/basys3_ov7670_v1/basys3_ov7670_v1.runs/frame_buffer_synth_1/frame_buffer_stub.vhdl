-- Copyright 1986-2018 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2018.1 (win64) Build 2188600 Wed Apr  4 18:40:38 MDT 2018
-- Date        : Tue Dec 18 12:10:47 2018
-- Host        : DESKTOP-010DTS5 running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode synth_stub
--               C:/Users/mcanb/OneDrive/Desktop/basys3_ov7670_v1/basys3_ov7670_v1/basys3_ov7670_v1.runs/frame_buffer_synth_1/frame_buffer_stub.vhdl
-- Design      : frame_buffer
-- Purpose     : Stub declaration of top-level module interface
-- Device      : xc7a35tcpg236-3
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity frame_buffer is
  Port ( 
    clka : in STD_LOGIC;
    wea : in STD_LOGIC_VECTOR ( 0 to 0 );
    addra : in STD_LOGIC_VECTOR ( 16 downto 0 );
    dina : in STD_LOGIC_VECTOR ( 11 downto 0 );
    clkb : in STD_LOGIC;
    addrb : in STD_LOGIC_VECTOR ( 16 downto 0 );
    doutb : out STD_LOGIC_VECTOR ( 11 downto 0 )
  );

end frame_buffer;

architecture stub of frame_buffer is
attribute syn_black_box : boolean;
attribute black_box_pad_pin : string;
attribute syn_black_box of stub : architecture is true;
attribute black_box_pad_pin of stub : architecture is "clka,wea[0:0],addra[16:0],dina[11:0],clkb,addrb[16:0],doutb[11:0]";
attribute x_core_info : string;
attribute x_core_info of stub : architecture is "blk_mem_gen_v8_4_1,Vivado 2018.1";
begin
end;
