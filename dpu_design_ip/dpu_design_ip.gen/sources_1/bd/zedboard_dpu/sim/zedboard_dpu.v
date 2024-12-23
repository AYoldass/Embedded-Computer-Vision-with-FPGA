//Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2020.1 (win64) Build 2902540 Wed May 27 19:54:49 MDT 2020
//Date        : Sun Dec 22 15:34:08 2024
//Host        : DESKTOP-MU42VMQ running 64-bit major release  (build 9200)
//Command     : generate_target zedboard_dpu.bd
//Design      : zedboard_dpu
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

(* CORE_GENERATION_INFO = "zedboard_dpu,IP_Integrator,{x_ipVendor=xilinx.com,x_ipLibrary=BlockDiagram,x_ipName=zedboard_dpu,x_ipVersion=1.00.a,x_ipLanguage=VERILOG,numBlks=7,numReposBlks=7,numNonXlnxBlks=0,numHierBlks=0,maxHierDepth=0,numSysgenBlks=0,numHlsBlks=0,numHdlrefBlks=0,numPkgbdBlks=0,bdsource=USER,da_ps7_cnt=1,synth_mode=OOC_per_IP}" *) (* HW_HANDOFF = "zedboard_dpu.hwdef" *) 
module zedboard_dpu
   (DDR_addr,
    DDR_ba,
    DDR_cas_n,
    DDR_ck_n,
    DDR_ck_p,
    DDR_cke,
    DDR_cs_n,
    DDR_dm,
    DDR_dq,
    DDR_dqs_n,
    DDR_dqs_p,
    DDR_odt,
    DDR_ras_n,
    DDR_reset_n,
    DDR_we_n,
    FIXED_IO_ddr_vrn,
    FIXED_IO_ddr_vrp,
    FIXED_IO_mio,
    FIXED_IO_ps_clk,
    FIXED_IO_ps_porb,
    FIXED_IO_ps_srstb);
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR ADDR" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DDR, AXI_ARBITRATION_SCHEME TDM, BURST_LENGTH 8, CAN_DEBUG false, CAS_LATENCY 11, CAS_WRITE_LATENCY 11, CS_ENABLED true, DATA_MASK_ENABLED true, DATA_WIDTH 8, MEMORY_TYPE COMPONENTS, MEM_ADDR_MAP ROW_COLUMN_BANK, SLOT Single, TIMEPERIOD_PS 1250" *) inout [14:0]DDR_addr;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR BA" *) inout [2:0]DDR_ba;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR CAS_N" *) inout DDR_cas_n;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR CK_N" *) inout DDR_ck_n;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR CK_P" *) inout DDR_ck_p;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR CKE" *) inout DDR_cke;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR CS_N" *) inout DDR_cs_n;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR DM" *) inout [3:0]DDR_dm;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR DQ" *) inout [31:0]DDR_dq;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR DQS_N" *) inout [3:0]DDR_dqs_n;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR DQS_P" *) inout [3:0]DDR_dqs_p;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR ODT" *) inout DDR_odt;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR RAS_N" *) inout DDR_ras_n;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR RESET_N" *) inout DDR_reset_n;
  (* X_INTERFACE_INFO = "xilinx.com:interface:ddrx:1.0 DDR WE_N" *) inout DDR_we_n;
  (* X_INTERFACE_INFO = "xilinx.com:display_processing_system7:fixedio:1.0 FIXED_IO DDR_VRN" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME FIXED_IO, CAN_DEBUG false" *) inout FIXED_IO_ddr_vrn;
  (* X_INTERFACE_INFO = "xilinx.com:display_processing_system7:fixedio:1.0 FIXED_IO DDR_VRP" *) inout FIXED_IO_ddr_vrp;
  (* X_INTERFACE_INFO = "xilinx.com:display_processing_system7:fixedio:1.0 FIXED_IO MIO" *) inout [53:0]FIXED_IO_mio;
  (* X_INTERFACE_INFO = "xilinx.com:display_processing_system7:fixedio:1.0 FIXED_IO PS_CLK" *) inout FIXED_IO_ps_clk;
  (* X_INTERFACE_INFO = "xilinx.com:display_processing_system7:fixedio:1.0 FIXED_IO PS_PORB" *) inout FIXED_IO_ps_porb;
  (* X_INTERFACE_INFO = "xilinx.com:display_processing_system7:fixedio:1.0 FIXED_IO PS_SRSTB" *) inout FIXED_IO_ps_srstb;

  wire Net;
  wire Net1;
  wire clk_wiz_0_clk_out150M;
  wire clk_wiz_0_clk_out300M;
  wire clk_wiz_0_locked;
  wire [39:0]dpu_eu_0_DPU0_M_AXI_DATA0_ARADDR;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_DATA0_ARBURST;
  wire [3:0]dpu_eu_0_DPU0_M_AXI_DATA0_ARCACHE;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_DATA0_ARID;
  wire [3:0]dpu_eu_0_DPU0_M_AXI_DATA0_ARLEN;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_DATA0_ARLOCK;
  wire [2:0]dpu_eu_0_DPU0_M_AXI_DATA0_ARPROT;
  wire [3:0]dpu_eu_0_DPU0_M_AXI_DATA0_ARQOS;
  wire dpu_eu_0_DPU0_M_AXI_DATA0_ARREADY;
  wire [2:0]dpu_eu_0_DPU0_M_AXI_DATA0_ARSIZE;
  wire dpu_eu_0_DPU0_M_AXI_DATA0_ARVALID;
  wire [39:0]dpu_eu_0_DPU0_M_AXI_DATA0_AWADDR;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_DATA0_AWBURST;
  wire [3:0]dpu_eu_0_DPU0_M_AXI_DATA0_AWCACHE;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_DATA0_AWID;
  wire [3:0]dpu_eu_0_DPU0_M_AXI_DATA0_AWLEN;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_DATA0_AWLOCK;
  wire [2:0]dpu_eu_0_DPU0_M_AXI_DATA0_AWPROT;
  wire [3:0]dpu_eu_0_DPU0_M_AXI_DATA0_AWQOS;
  wire dpu_eu_0_DPU0_M_AXI_DATA0_AWREADY;
  wire [2:0]dpu_eu_0_DPU0_M_AXI_DATA0_AWSIZE;
  wire dpu_eu_0_DPU0_M_AXI_DATA0_AWVALID;
  wire [5:0]dpu_eu_0_DPU0_M_AXI_DATA0_BID;
  wire dpu_eu_0_DPU0_M_AXI_DATA0_BREADY;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_DATA0_BRESP;
  wire dpu_eu_0_DPU0_M_AXI_DATA0_BVALID;
  wire [63:0]dpu_eu_0_DPU0_M_AXI_DATA0_RDATA;
  wire [5:0]dpu_eu_0_DPU0_M_AXI_DATA0_RID;
  wire dpu_eu_0_DPU0_M_AXI_DATA0_RLAST;
  wire dpu_eu_0_DPU0_M_AXI_DATA0_RREADY;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_DATA0_RRESP;
  wire dpu_eu_0_DPU0_M_AXI_DATA0_RVALID;
  wire [63:0]dpu_eu_0_DPU0_M_AXI_DATA0_WDATA;
  wire [5:0]dpu_eu_0_DPU0_M_AXI_DATA0_WID;
  wire dpu_eu_0_DPU0_M_AXI_DATA0_WLAST;
  wire dpu_eu_0_DPU0_M_AXI_DATA0_WREADY;
  wire [7:0]dpu_eu_0_DPU0_M_AXI_DATA0_WSTRB;
  wire dpu_eu_0_DPU0_M_AXI_DATA0_WVALID;
  wire [39:0]dpu_eu_0_DPU0_M_AXI_DATA1_ARADDR;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_DATA1_ARBURST;
  wire [3:0]dpu_eu_0_DPU0_M_AXI_DATA1_ARCACHE;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_DATA1_ARID;
  wire [3:0]dpu_eu_0_DPU0_M_AXI_DATA1_ARLEN;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_DATA1_ARLOCK;
  wire [2:0]dpu_eu_0_DPU0_M_AXI_DATA1_ARPROT;
  wire [3:0]dpu_eu_0_DPU0_M_AXI_DATA1_ARQOS;
  wire dpu_eu_0_DPU0_M_AXI_DATA1_ARREADY;
  wire [2:0]dpu_eu_0_DPU0_M_AXI_DATA1_ARSIZE;
  wire dpu_eu_0_DPU0_M_AXI_DATA1_ARVALID;
  wire [39:0]dpu_eu_0_DPU0_M_AXI_DATA1_AWADDR;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_DATA1_AWBURST;
  wire [3:0]dpu_eu_0_DPU0_M_AXI_DATA1_AWCACHE;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_DATA1_AWID;
  wire [3:0]dpu_eu_0_DPU0_M_AXI_DATA1_AWLEN;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_DATA1_AWLOCK;
  wire [2:0]dpu_eu_0_DPU0_M_AXI_DATA1_AWPROT;
  wire [3:0]dpu_eu_0_DPU0_M_AXI_DATA1_AWQOS;
  wire dpu_eu_0_DPU0_M_AXI_DATA1_AWREADY;
  wire [2:0]dpu_eu_0_DPU0_M_AXI_DATA1_AWSIZE;
  wire dpu_eu_0_DPU0_M_AXI_DATA1_AWVALID;
  wire [5:0]dpu_eu_0_DPU0_M_AXI_DATA1_BID;
  wire dpu_eu_0_DPU0_M_AXI_DATA1_BREADY;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_DATA1_BRESP;
  wire dpu_eu_0_DPU0_M_AXI_DATA1_BVALID;
  wire [63:0]dpu_eu_0_DPU0_M_AXI_DATA1_RDATA;
  wire [5:0]dpu_eu_0_DPU0_M_AXI_DATA1_RID;
  wire dpu_eu_0_DPU0_M_AXI_DATA1_RLAST;
  wire dpu_eu_0_DPU0_M_AXI_DATA1_RREADY;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_DATA1_RRESP;
  wire dpu_eu_0_DPU0_M_AXI_DATA1_RVALID;
  wire [63:0]dpu_eu_0_DPU0_M_AXI_DATA1_WDATA;
  wire [5:0]dpu_eu_0_DPU0_M_AXI_DATA1_WID;
  wire dpu_eu_0_DPU0_M_AXI_DATA1_WLAST;
  wire dpu_eu_0_DPU0_M_AXI_DATA1_WREADY;
  wire [7:0]dpu_eu_0_DPU0_M_AXI_DATA1_WSTRB;
  wire dpu_eu_0_DPU0_M_AXI_DATA1_WVALID;
  wire [39:0]dpu_eu_0_DPU0_M_AXI_INSTR_ARADDR;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_INSTR_ARBURST;
  wire [3:0]dpu_eu_0_DPU0_M_AXI_INSTR_ARCACHE;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_INSTR_ARID;
  wire [3:0]dpu_eu_0_DPU0_M_AXI_INSTR_ARLEN;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_INSTR_ARLOCK;
  wire [2:0]dpu_eu_0_DPU0_M_AXI_INSTR_ARPROT;
  wire [3:0]dpu_eu_0_DPU0_M_AXI_INSTR_ARQOS;
  wire dpu_eu_0_DPU0_M_AXI_INSTR_ARREADY;
  wire [2:0]dpu_eu_0_DPU0_M_AXI_INSTR_ARSIZE;
  wire dpu_eu_0_DPU0_M_AXI_INSTR_ARVALID;
  wire [39:0]dpu_eu_0_DPU0_M_AXI_INSTR_AWADDR;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_INSTR_AWBURST;
  wire [3:0]dpu_eu_0_DPU0_M_AXI_INSTR_AWCACHE;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_INSTR_AWID;
  wire [3:0]dpu_eu_0_DPU0_M_AXI_INSTR_AWLEN;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_INSTR_AWLOCK;
  wire [2:0]dpu_eu_0_DPU0_M_AXI_INSTR_AWPROT;
  wire [3:0]dpu_eu_0_DPU0_M_AXI_INSTR_AWQOS;
  wire dpu_eu_0_DPU0_M_AXI_INSTR_AWREADY;
  wire [2:0]dpu_eu_0_DPU0_M_AXI_INSTR_AWSIZE;
  wire dpu_eu_0_DPU0_M_AXI_INSTR_AWVALID;
  wire [5:0]dpu_eu_0_DPU0_M_AXI_INSTR_BID;
  wire dpu_eu_0_DPU0_M_AXI_INSTR_BREADY;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_INSTR_BRESP;
  wire dpu_eu_0_DPU0_M_AXI_INSTR_BVALID;
  wire [31:0]dpu_eu_0_DPU0_M_AXI_INSTR_RDATA;
  wire [5:0]dpu_eu_0_DPU0_M_AXI_INSTR_RID;
  wire dpu_eu_0_DPU0_M_AXI_INSTR_RLAST;
  wire dpu_eu_0_DPU0_M_AXI_INSTR_RREADY;
  wire [1:0]dpu_eu_0_DPU0_M_AXI_INSTR_RRESP;
  wire dpu_eu_0_DPU0_M_AXI_INSTR_RVALID;
  wire [31:0]dpu_eu_0_DPU0_M_AXI_INSTR_WDATA;
  wire [5:0]dpu_eu_0_DPU0_M_AXI_INSTR_WID;
  wire dpu_eu_0_DPU0_M_AXI_INSTR_WLAST;
  wire dpu_eu_0_DPU0_M_AXI_INSTR_WREADY;
  wire [3:0]dpu_eu_0_DPU0_M_AXI_INSTR_WSTRB;
  wire dpu_eu_0_DPU0_M_AXI_INSTR_WVALID;
  wire [0:0]dpu_eu_0_dpu_interrupt;
  wire [0:0]proc_sys_reset_0_peripheral_aresetn;
  wire [0:0]proc_sys_reset_150M_peripheral_aresetn;
  wire [0:0]proc_sys_reset_300M_peripheral_aresetn;
  wire [14:0]processing_system7_0_DDR_ADDR;
  wire [2:0]processing_system7_0_DDR_BA;
  wire processing_system7_0_DDR_CAS_N;
  wire processing_system7_0_DDR_CKE;
  wire processing_system7_0_DDR_CK_N;
  wire processing_system7_0_DDR_CK_P;
  wire processing_system7_0_DDR_CS_N;
  wire [3:0]processing_system7_0_DDR_DM;
  wire [31:0]processing_system7_0_DDR_DQ;
  wire [3:0]processing_system7_0_DDR_DQS_N;
  wire [3:0]processing_system7_0_DDR_DQS_P;
  wire processing_system7_0_DDR_ODT;
  wire processing_system7_0_DDR_RAS_N;
  wire processing_system7_0_DDR_RESET_N;
  wire processing_system7_0_DDR_WE_N;
  wire processing_system7_0_FIXED_IO_DDR_VRN;
  wire processing_system7_0_FIXED_IO_DDR_VRP;
  wire [53:0]processing_system7_0_FIXED_IO_MIO;
  wire processing_system7_0_FIXED_IO_PS_CLK;
  wire processing_system7_0_FIXED_IO_PS_PORB;
  wire processing_system7_0_FIXED_IO_PS_SRSTB;
  wire [31:0]processing_system7_0_M_AXI_GP0_ARADDR;
  wire [1:0]processing_system7_0_M_AXI_GP0_ARBURST;
  wire [3:0]processing_system7_0_M_AXI_GP0_ARCACHE;
  wire [11:0]processing_system7_0_M_AXI_GP0_ARID;
  wire [3:0]processing_system7_0_M_AXI_GP0_ARLEN;
  wire [1:0]processing_system7_0_M_AXI_GP0_ARLOCK;
  wire [2:0]processing_system7_0_M_AXI_GP0_ARPROT;
  wire [3:0]processing_system7_0_M_AXI_GP0_ARQOS;
  wire processing_system7_0_M_AXI_GP0_ARREADY;
  wire [2:0]processing_system7_0_M_AXI_GP0_ARSIZE;
  wire processing_system7_0_M_AXI_GP0_ARVALID;
  wire [31:0]processing_system7_0_M_AXI_GP0_AWADDR;
  wire [1:0]processing_system7_0_M_AXI_GP0_AWBURST;
  wire [3:0]processing_system7_0_M_AXI_GP0_AWCACHE;
  wire [11:0]processing_system7_0_M_AXI_GP0_AWID;
  wire [3:0]processing_system7_0_M_AXI_GP0_AWLEN;
  wire [1:0]processing_system7_0_M_AXI_GP0_AWLOCK;
  wire [2:0]processing_system7_0_M_AXI_GP0_AWPROT;
  wire [3:0]processing_system7_0_M_AXI_GP0_AWQOS;
  wire processing_system7_0_M_AXI_GP0_AWREADY;
  wire [2:0]processing_system7_0_M_AXI_GP0_AWSIZE;
  wire processing_system7_0_M_AXI_GP0_AWVALID;
  wire [11:0]processing_system7_0_M_AXI_GP0_BID;
  wire processing_system7_0_M_AXI_GP0_BREADY;
  wire [1:0]processing_system7_0_M_AXI_GP0_BRESP;
  wire processing_system7_0_M_AXI_GP0_BVALID;
  wire [31:0]processing_system7_0_M_AXI_GP0_RDATA;
  wire [11:0]processing_system7_0_M_AXI_GP0_RID;
  wire processing_system7_0_M_AXI_GP0_RLAST;
  wire processing_system7_0_M_AXI_GP0_RREADY;
  wire [1:0]processing_system7_0_M_AXI_GP0_RRESP;
  wire processing_system7_0_M_AXI_GP0_RVALID;
  wire [31:0]processing_system7_0_M_AXI_GP0_WDATA;
  wire [11:0]processing_system7_0_M_AXI_GP0_WID;
  wire processing_system7_0_M_AXI_GP0_WLAST;
  wire processing_system7_0_M_AXI_GP0_WREADY;
  wire [3:0]processing_system7_0_M_AXI_GP0_WSTRB;
  wire processing_system7_0_M_AXI_GP0_WVALID;
  wire [1:0]xlconcat_0_dout;

  zedboard_dpu_clk_wiz_0_0 clk_wiz_0
       (.clk_in1(Net),
        .clk_out150M(clk_wiz_0_clk_out150M),
        .clk_out300M(clk_wiz_0_clk_out300M),
        .locked(clk_wiz_0_locked),
        .resetn(Net1));
  zedboard_dpu_dpu_eu_0_0 dpu_eu_0
       (.dpu0_m_axi_data0_araddr(dpu_eu_0_DPU0_M_AXI_DATA0_ARADDR),
        .dpu0_m_axi_data0_arburst(dpu_eu_0_DPU0_M_AXI_DATA0_ARBURST),
        .dpu0_m_axi_data0_arcache(dpu_eu_0_DPU0_M_AXI_DATA0_ARCACHE),
        .dpu0_m_axi_data0_arid(dpu_eu_0_DPU0_M_AXI_DATA0_ARID),
        .dpu0_m_axi_data0_arlen(dpu_eu_0_DPU0_M_AXI_DATA0_ARLEN),
        .dpu0_m_axi_data0_arlock(dpu_eu_0_DPU0_M_AXI_DATA0_ARLOCK),
        .dpu0_m_axi_data0_arprot(dpu_eu_0_DPU0_M_AXI_DATA0_ARPROT),
        .dpu0_m_axi_data0_arqos(dpu_eu_0_DPU0_M_AXI_DATA0_ARQOS),
        .dpu0_m_axi_data0_arready(dpu_eu_0_DPU0_M_AXI_DATA0_ARREADY),
        .dpu0_m_axi_data0_arsize(dpu_eu_0_DPU0_M_AXI_DATA0_ARSIZE),
        .dpu0_m_axi_data0_arvalid(dpu_eu_0_DPU0_M_AXI_DATA0_ARVALID),
        .dpu0_m_axi_data0_awaddr(dpu_eu_0_DPU0_M_AXI_DATA0_AWADDR),
        .dpu0_m_axi_data0_awburst(dpu_eu_0_DPU0_M_AXI_DATA0_AWBURST),
        .dpu0_m_axi_data0_awcache(dpu_eu_0_DPU0_M_AXI_DATA0_AWCACHE),
        .dpu0_m_axi_data0_awid(dpu_eu_0_DPU0_M_AXI_DATA0_AWID),
        .dpu0_m_axi_data0_awlen(dpu_eu_0_DPU0_M_AXI_DATA0_AWLEN),
        .dpu0_m_axi_data0_awlock(dpu_eu_0_DPU0_M_AXI_DATA0_AWLOCK),
        .dpu0_m_axi_data0_awprot(dpu_eu_0_DPU0_M_AXI_DATA0_AWPROT),
        .dpu0_m_axi_data0_awqos(dpu_eu_0_DPU0_M_AXI_DATA0_AWQOS),
        .dpu0_m_axi_data0_awready(dpu_eu_0_DPU0_M_AXI_DATA0_AWREADY),
        .dpu0_m_axi_data0_awsize(dpu_eu_0_DPU0_M_AXI_DATA0_AWSIZE),
        .dpu0_m_axi_data0_awvalid(dpu_eu_0_DPU0_M_AXI_DATA0_AWVALID),
        .dpu0_m_axi_data0_bid(dpu_eu_0_DPU0_M_AXI_DATA0_BID),
        .dpu0_m_axi_data0_bready(dpu_eu_0_DPU0_M_AXI_DATA0_BREADY),
        .dpu0_m_axi_data0_bresp(dpu_eu_0_DPU0_M_AXI_DATA0_BRESP),
        .dpu0_m_axi_data0_bvalid(dpu_eu_0_DPU0_M_AXI_DATA0_BVALID),
        .dpu0_m_axi_data0_rdata(dpu_eu_0_DPU0_M_AXI_DATA0_RDATA),
        .dpu0_m_axi_data0_rid(dpu_eu_0_DPU0_M_AXI_DATA0_RID),
        .dpu0_m_axi_data0_rlast(dpu_eu_0_DPU0_M_AXI_DATA0_RLAST),
        .dpu0_m_axi_data0_rready(dpu_eu_0_DPU0_M_AXI_DATA0_RREADY),
        .dpu0_m_axi_data0_rresp(dpu_eu_0_DPU0_M_AXI_DATA0_RRESP),
        .dpu0_m_axi_data0_rvalid(dpu_eu_0_DPU0_M_AXI_DATA0_RVALID),
        .dpu0_m_axi_data0_wdata(dpu_eu_0_DPU0_M_AXI_DATA0_WDATA),
        .dpu0_m_axi_data0_wid(dpu_eu_0_DPU0_M_AXI_DATA0_WID),
        .dpu0_m_axi_data0_wlast(dpu_eu_0_DPU0_M_AXI_DATA0_WLAST),
        .dpu0_m_axi_data0_wready(dpu_eu_0_DPU0_M_AXI_DATA0_WREADY),
        .dpu0_m_axi_data0_wstrb(dpu_eu_0_DPU0_M_AXI_DATA0_WSTRB),
        .dpu0_m_axi_data0_wvalid(dpu_eu_0_DPU0_M_AXI_DATA0_WVALID),
        .dpu0_m_axi_data1_araddr(dpu_eu_0_DPU0_M_AXI_DATA1_ARADDR),
        .dpu0_m_axi_data1_arburst(dpu_eu_0_DPU0_M_AXI_DATA1_ARBURST),
        .dpu0_m_axi_data1_arcache(dpu_eu_0_DPU0_M_AXI_DATA1_ARCACHE),
        .dpu0_m_axi_data1_arid(dpu_eu_0_DPU0_M_AXI_DATA1_ARID),
        .dpu0_m_axi_data1_arlen(dpu_eu_0_DPU0_M_AXI_DATA1_ARLEN),
        .dpu0_m_axi_data1_arlock(dpu_eu_0_DPU0_M_AXI_DATA1_ARLOCK),
        .dpu0_m_axi_data1_arprot(dpu_eu_0_DPU0_M_AXI_DATA1_ARPROT),
        .dpu0_m_axi_data1_arqos(dpu_eu_0_DPU0_M_AXI_DATA1_ARQOS),
        .dpu0_m_axi_data1_arready(dpu_eu_0_DPU0_M_AXI_DATA1_ARREADY),
        .dpu0_m_axi_data1_arsize(dpu_eu_0_DPU0_M_AXI_DATA1_ARSIZE),
        .dpu0_m_axi_data1_arvalid(dpu_eu_0_DPU0_M_AXI_DATA1_ARVALID),
        .dpu0_m_axi_data1_awaddr(dpu_eu_0_DPU0_M_AXI_DATA1_AWADDR),
        .dpu0_m_axi_data1_awburst(dpu_eu_0_DPU0_M_AXI_DATA1_AWBURST),
        .dpu0_m_axi_data1_awcache(dpu_eu_0_DPU0_M_AXI_DATA1_AWCACHE),
        .dpu0_m_axi_data1_awid(dpu_eu_0_DPU0_M_AXI_DATA1_AWID),
        .dpu0_m_axi_data1_awlen(dpu_eu_0_DPU0_M_AXI_DATA1_AWLEN),
        .dpu0_m_axi_data1_awlock(dpu_eu_0_DPU0_M_AXI_DATA1_AWLOCK),
        .dpu0_m_axi_data1_awprot(dpu_eu_0_DPU0_M_AXI_DATA1_AWPROT),
        .dpu0_m_axi_data1_awqos(dpu_eu_0_DPU0_M_AXI_DATA1_AWQOS),
        .dpu0_m_axi_data1_awready(dpu_eu_0_DPU0_M_AXI_DATA1_AWREADY),
        .dpu0_m_axi_data1_awsize(dpu_eu_0_DPU0_M_AXI_DATA1_AWSIZE),
        .dpu0_m_axi_data1_awvalid(dpu_eu_0_DPU0_M_AXI_DATA1_AWVALID),
        .dpu0_m_axi_data1_bid(dpu_eu_0_DPU0_M_AXI_DATA1_BID),
        .dpu0_m_axi_data1_bready(dpu_eu_0_DPU0_M_AXI_DATA1_BREADY),
        .dpu0_m_axi_data1_bresp(dpu_eu_0_DPU0_M_AXI_DATA1_BRESP),
        .dpu0_m_axi_data1_bvalid(dpu_eu_0_DPU0_M_AXI_DATA1_BVALID),
        .dpu0_m_axi_data1_rdata(dpu_eu_0_DPU0_M_AXI_DATA1_RDATA),
        .dpu0_m_axi_data1_rid(dpu_eu_0_DPU0_M_AXI_DATA1_RID),
        .dpu0_m_axi_data1_rlast(dpu_eu_0_DPU0_M_AXI_DATA1_RLAST),
        .dpu0_m_axi_data1_rready(dpu_eu_0_DPU0_M_AXI_DATA1_RREADY),
        .dpu0_m_axi_data1_rresp(dpu_eu_0_DPU0_M_AXI_DATA1_RRESP),
        .dpu0_m_axi_data1_rvalid(dpu_eu_0_DPU0_M_AXI_DATA1_RVALID),
        .dpu0_m_axi_data1_wdata(dpu_eu_0_DPU0_M_AXI_DATA1_WDATA),
        .dpu0_m_axi_data1_wid(dpu_eu_0_DPU0_M_AXI_DATA1_WID),
        .dpu0_m_axi_data1_wlast(dpu_eu_0_DPU0_M_AXI_DATA1_WLAST),
        .dpu0_m_axi_data1_wready(dpu_eu_0_DPU0_M_AXI_DATA1_WREADY),
        .dpu0_m_axi_data1_wstrb(dpu_eu_0_DPU0_M_AXI_DATA1_WSTRB),
        .dpu0_m_axi_data1_wvalid(dpu_eu_0_DPU0_M_AXI_DATA1_WVALID),
        .dpu0_m_axi_instr_araddr(dpu_eu_0_DPU0_M_AXI_INSTR_ARADDR),
        .dpu0_m_axi_instr_arburst(dpu_eu_0_DPU0_M_AXI_INSTR_ARBURST),
        .dpu0_m_axi_instr_arcache(dpu_eu_0_DPU0_M_AXI_INSTR_ARCACHE),
        .dpu0_m_axi_instr_arid(dpu_eu_0_DPU0_M_AXI_INSTR_ARID),
        .dpu0_m_axi_instr_arlen(dpu_eu_0_DPU0_M_AXI_INSTR_ARLEN),
        .dpu0_m_axi_instr_arlock(dpu_eu_0_DPU0_M_AXI_INSTR_ARLOCK),
        .dpu0_m_axi_instr_arprot(dpu_eu_0_DPU0_M_AXI_INSTR_ARPROT),
        .dpu0_m_axi_instr_arqos(dpu_eu_0_DPU0_M_AXI_INSTR_ARQOS),
        .dpu0_m_axi_instr_arready(dpu_eu_0_DPU0_M_AXI_INSTR_ARREADY),
        .dpu0_m_axi_instr_arsize(dpu_eu_0_DPU0_M_AXI_INSTR_ARSIZE),
        .dpu0_m_axi_instr_arvalid(dpu_eu_0_DPU0_M_AXI_INSTR_ARVALID),
        .dpu0_m_axi_instr_awaddr(dpu_eu_0_DPU0_M_AXI_INSTR_AWADDR),
        .dpu0_m_axi_instr_awburst(dpu_eu_0_DPU0_M_AXI_INSTR_AWBURST),
        .dpu0_m_axi_instr_awcache(dpu_eu_0_DPU0_M_AXI_INSTR_AWCACHE),
        .dpu0_m_axi_instr_awid(dpu_eu_0_DPU0_M_AXI_INSTR_AWID),
        .dpu0_m_axi_instr_awlen(dpu_eu_0_DPU0_M_AXI_INSTR_AWLEN),
        .dpu0_m_axi_instr_awlock(dpu_eu_0_DPU0_M_AXI_INSTR_AWLOCK),
        .dpu0_m_axi_instr_awprot(dpu_eu_0_DPU0_M_AXI_INSTR_AWPROT),
        .dpu0_m_axi_instr_awqos(dpu_eu_0_DPU0_M_AXI_INSTR_AWQOS),
        .dpu0_m_axi_instr_awready(dpu_eu_0_DPU0_M_AXI_INSTR_AWREADY),
        .dpu0_m_axi_instr_awsize(dpu_eu_0_DPU0_M_AXI_INSTR_AWSIZE),
        .dpu0_m_axi_instr_awvalid(dpu_eu_0_DPU0_M_AXI_INSTR_AWVALID),
        .dpu0_m_axi_instr_bid(dpu_eu_0_DPU0_M_AXI_INSTR_BID),
        .dpu0_m_axi_instr_bready(dpu_eu_0_DPU0_M_AXI_INSTR_BREADY),
        .dpu0_m_axi_instr_bresp(dpu_eu_0_DPU0_M_AXI_INSTR_BRESP),
        .dpu0_m_axi_instr_bvalid(dpu_eu_0_DPU0_M_AXI_INSTR_BVALID),
        .dpu0_m_axi_instr_rdata(dpu_eu_0_DPU0_M_AXI_INSTR_RDATA),
        .dpu0_m_axi_instr_rid(dpu_eu_0_DPU0_M_AXI_INSTR_RID),
        .dpu0_m_axi_instr_rlast(dpu_eu_0_DPU0_M_AXI_INSTR_RLAST),
        .dpu0_m_axi_instr_rready(dpu_eu_0_DPU0_M_AXI_INSTR_RREADY),
        .dpu0_m_axi_instr_rresp(dpu_eu_0_DPU0_M_AXI_INSTR_RRESP),
        .dpu0_m_axi_instr_rvalid(dpu_eu_0_DPU0_M_AXI_INSTR_RVALID),
        .dpu0_m_axi_instr_wdata(dpu_eu_0_DPU0_M_AXI_INSTR_WDATA),
        .dpu0_m_axi_instr_wid(dpu_eu_0_DPU0_M_AXI_INSTR_WID),
        .dpu0_m_axi_instr_wlast(dpu_eu_0_DPU0_M_AXI_INSTR_WLAST),
        .dpu0_m_axi_instr_wready(dpu_eu_0_DPU0_M_AXI_INSTR_WREADY),
        .dpu0_m_axi_instr_wstrb(dpu_eu_0_DPU0_M_AXI_INSTR_WSTRB),
        .dpu0_m_axi_instr_wvalid(dpu_eu_0_DPU0_M_AXI_INSTR_WVALID),
        .dpu_2x_clk(clk_wiz_0_clk_out300M),
        .dpu_2x_resetn(proc_sys_reset_300M_peripheral_aresetn),
        .dpu_interrupt(dpu_eu_0_dpu_interrupt),
        .m_axi_dpu_aclk(clk_wiz_0_clk_out150M),
        .m_axi_dpu_aresetn(proc_sys_reset_150M_peripheral_aresetn),
        .s_axi_aclk(Net),
        .s_axi_araddr(processing_system7_0_M_AXI_GP0_ARADDR),
        .s_axi_arburst(processing_system7_0_M_AXI_GP0_ARBURST),
        .s_axi_arcache(processing_system7_0_M_AXI_GP0_ARCACHE),
        .s_axi_aresetn(proc_sys_reset_0_peripheral_aresetn),
        .s_axi_arid(processing_system7_0_M_AXI_GP0_ARID),
        .s_axi_arlen(processing_system7_0_M_AXI_GP0_ARLEN),
        .s_axi_arlock(processing_system7_0_M_AXI_GP0_ARLOCK),
        .s_axi_arprot(processing_system7_0_M_AXI_GP0_ARPROT),
        .s_axi_arqos(processing_system7_0_M_AXI_GP0_ARQOS),
        .s_axi_arready(processing_system7_0_M_AXI_GP0_ARREADY),
        .s_axi_arregion({1'b0,1'b0,1'b0,1'b0}),
        .s_axi_arsize(processing_system7_0_M_AXI_GP0_ARSIZE),
        .s_axi_arvalid(processing_system7_0_M_AXI_GP0_ARVALID),
        .s_axi_awaddr(processing_system7_0_M_AXI_GP0_AWADDR),
        .s_axi_awburst(processing_system7_0_M_AXI_GP0_AWBURST),
        .s_axi_awcache(processing_system7_0_M_AXI_GP0_AWCACHE),
        .s_axi_awid(processing_system7_0_M_AXI_GP0_AWID),
        .s_axi_awlen(processing_system7_0_M_AXI_GP0_AWLEN),
        .s_axi_awlock(processing_system7_0_M_AXI_GP0_AWLOCK),
        .s_axi_awprot(processing_system7_0_M_AXI_GP0_AWPROT),
        .s_axi_awqos(processing_system7_0_M_AXI_GP0_AWQOS),
        .s_axi_awready(processing_system7_0_M_AXI_GP0_AWREADY),
        .s_axi_awregion({1'b0,1'b0,1'b0,1'b0}),
        .s_axi_awsize(processing_system7_0_M_AXI_GP0_AWSIZE),
        .s_axi_awvalid(processing_system7_0_M_AXI_GP0_AWVALID),
        .s_axi_bid(processing_system7_0_M_AXI_GP0_BID),
        .s_axi_bready(processing_system7_0_M_AXI_GP0_BREADY),
        .s_axi_bresp(processing_system7_0_M_AXI_GP0_BRESP),
        .s_axi_bvalid(processing_system7_0_M_AXI_GP0_BVALID),
        .s_axi_rdata(processing_system7_0_M_AXI_GP0_RDATA),
        .s_axi_rid(processing_system7_0_M_AXI_GP0_RID),
        .s_axi_rlast(processing_system7_0_M_AXI_GP0_RLAST),
        .s_axi_rready(processing_system7_0_M_AXI_GP0_RREADY),
        .s_axi_rresp(processing_system7_0_M_AXI_GP0_RRESP),
        .s_axi_rvalid(processing_system7_0_M_AXI_GP0_RVALID),
        .s_axi_wdata(processing_system7_0_M_AXI_GP0_WDATA),
        .s_axi_wid(processing_system7_0_M_AXI_GP0_WID),
        .s_axi_wlast(processing_system7_0_M_AXI_GP0_WLAST),
        .s_axi_wready(processing_system7_0_M_AXI_GP0_WREADY),
        .s_axi_wstrb(processing_system7_0_M_AXI_GP0_WSTRB),
        .s_axi_wvalid(processing_system7_0_M_AXI_GP0_WVALID));
  zedboard_dpu_proc_sys_reset_0_2 proc_sys_reset_0
       (.aux_reset_in(1'b1),
        .dcm_locked(1'b1),
        .ext_reset_in(Net1),
        .mb_debug_sys_rst(1'b0),
        .peripheral_aresetn(proc_sys_reset_0_peripheral_aresetn),
        .slowest_sync_clk(Net));
  zedboard_dpu_proc_sys_reset_0_1 proc_sys_reset_150M
       (.aux_reset_in(1'b1),
        .dcm_locked(clk_wiz_0_locked),
        .ext_reset_in(Net1),
        .mb_debug_sys_rst(1'b0),
        .peripheral_aresetn(proc_sys_reset_150M_peripheral_aresetn),
        .slowest_sync_clk(clk_wiz_0_clk_out150M));
  zedboard_dpu_proc_sys_reset_0_0 proc_sys_reset_300M
       (.aux_reset_in(1'b1),
        .dcm_locked(clk_wiz_0_locked),
        .ext_reset_in(Net1),
        .mb_debug_sys_rst(1'b0),
        .peripheral_aresetn(proc_sys_reset_300M_peripheral_aresetn),
        .slowest_sync_clk(clk_wiz_0_clk_out300M));
  zedboard_dpu_processing_system7_0_0 processing_system7_0
       (.DDR_Addr(DDR_addr[14:0]),
        .DDR_BankAddr(DDR_ba[2:0]),
        .DDR_CAS_n(DDR_cas_n),
        .DDR_CKE(DDR_cke),
        .DDR_CS_n(DDR_cs_n),
        .DDR_Clk(DDR_ck_p),
        .DDR_Clk_n(DDR_ck_n),
        .DDR_DM(DDR_dm[3:0]),
        .DDR_DQ(DDR_dq[31:0]),
        .DDR_DQS(DDR_dqs_p[3:0]),
        .DDR_DQS_n(DDR_dqs_n[3:0]),
        .DDR_DRSTB(DDR_reset_n),
        .DDR_ODT(DDR_odt),
        .DDR_RAS_n(DDR_ras_n),
        .DDR_VRN(FIXED_IO_ddr_vrn),
        .DDR_VRP(FIXED_IO_ddr_vrp),
        .DDR_WEB(DDR_we_n),
        .FCLK_CLK0(Net),
        .FCLK_RESET0_N(Net1),
        .IRQ_F2P(xlconcat_0_dout),
        .MIO(FIXED_IO_mio[53:0]),
        .M_AXI_GP0_ACLK(Net),
        .M_AXI_GP0_ARADDR(processing_system7_0_M_AXI_GP0_ARADDR),
        .M_AXI_GP0_ARBURST(processing_system7_0_M_AXI_GP0_ARBURST),
        .M_AXI_GP0_ARCACHE(processing_system7_0_M_AXI_GP0_ARCACHE),
        .M_AXI_GP0_ARID(processing_system7_0_M_AXI_GP0_ARID),
        .M_AXI_GP0_ARLEN(processing_system7_0_M_AXI_GP0_ARLEN),
        .M_AXI_GP0_ARLOCK(processing_system7_0_M_AXI_GP0_ARLOCK),
        .M_AXI_GP0_ARPROT(processing_system7_0_M_AXI_GP0_ARPROT),
        .M_AXI_GP0_ARQOS(processing_system7_0_M_AXI_GP0_ARQOS),
        .M_AXI_GP0_ARREADY(processing_system7_0_M_AXI_GP0_ARREADY),
        .M_AXI_GP0_ARSIZE(processing_system7_0_M_AXI_GP0_ARSIZE),
        .M_AXI_GP0_ARVALID(processing_system7_0_M_AXI_GP0_ARVALID),
        .M_AXI_GP0_AWADDR(processing_system7_0_M_AXI_GP0_AWADDR),
        .M_AXI_GP0_AWBURST(processing_system7_0_M_AXI_GP0_AWBURST),
        .M_AXI_GP0_AWCACHE(processing_system7_0_M_AXI_GP0_AWCACHE),
        .M_AXI_GP0_AWID(processing_system7_0_M_AXI_GP0_AWID),
        .M_AXI_GP0_AWLEN(processing_system7_0_M_AXI_GP0_AWLEN),
        .M_AXI_GP0_AWLOCK(processing_system7_0_M_AXI_GP0_AWLOCK),
        .M_AXI_GP0_AWPROT(processing_system7_0_M_AXI_GP0_AWPROT),
        .M_AXI_GP0_AWQOS(processing_system7_0_M_AXI_GP0_AWQOS),
        .M_AXI_GP0_AWREADY(processing_system7_0_M_AXI_GP0_AWREADY),
        .M_AXI_GP0_AWSIZE(processing_system7_0_M_AXI_GP0_AWSIZE),
        .M_AXI_GP0_AWVALID(processing_system7_0_M_AXI_GP0_AWVALID),
        .M_AXI_GP0_BID(processing_system7_0_M_AXI_GP0_BID),
        .M_AXI_GP0_BREADY(processing_system7_0_M_AXI_GP0_BREADY),
        .M_AXI_GP0_BRESP(processing_system7_0_M_AXI_GP0_BRESP),
        .M_AXI_GP0_BVALID(processing_system7_0_M_AXI_GP0_BVALID),
        .M_AXI_GP0_RDATA(processing_system7_0_M_AXI_GP0_RDATA),
        .M_AXI_GP0_RID(processing_system7_0_M_AXI_GP0_RID),
        .M_AXI_GP0_RLAST(processing_system7_0_M_AXI_GP0_RLAST),
        .M_AXI_GP0_RREADY(processing_system7_0_M_AXI_GP0_RREADY),
        .M_AXI_GP0_RRESP(processing_system7_0_M_AXI_GP0_RRESP),
        .M_AXI_GP0_RVALID(processing_system7_0_M_AXI_GP0_RVALID),
        .M_AXI_GP0_WDATA(processing_system7_0_M_AXI_GP0_WDATA),
        .M_AXI_GP0_WID(processing_system7_0_M_AXI_GP0_WID),
        .M_AXI_GP0_WLAST(processing_system7_0_M_AXI_GP0_WLAST),
        .M_AXI_GP0_WREADY(processing_system7_0_M_AXI_GP0_WREADY),
        .M_AXI_GP0_WSTRB(processing_system7_0_M_AXI_GP0_WSTRB),
        .M_AXI_GP0_WVALID(processing_system7_0_M_AXI_GP0_WVALID),
        .PS_CLK(FIXED_IO_ps_clk),
        .PS_PORB(FIXED_IO_ps_porb),
        .PS_SRSTB(FIXED_IO_ps_srstb),
        .S_AXI_HP0_ACLK(clk_wiz_0_clk_out150M),
        .S_AXI_HP0_ARADDR(dpu_eu_0_DPU0_M_AXI_DATA0_ARADDR[31:0]),
        .S_AXI_HP0_ARBURST(dpu_eu_0_DPU0_M_AXI_DATA0_ARBURST),
        .S_AXI_HP0_ARCACHE(dpu_eu_0_DPU0_M_AXI_DATA0_ARCACHE),
        .S_AXI_HP0_ARID({1'b0,1'b0,1'b0,1'b0,dpu_eu_0_DPU0_M_AXI_DATA0_ARID}),
        .S_AXI_HP0_ARLEN(dpu_eu_0_DPU0_M_AXI_DATA0_ARLEN),
        .S_AXI_HP0_ARLOCK(dpu_eu_0_DPU0_M_AXI_DATA0_ARLOCK),
        .S_AXI_HP0_ARPROT(dpu_eu_0_DPU0_M_AXI_DATA0_ARPROT),
        .S_AXI_HP0_ARQOS(dpu_eu_0_DPU0_M_AXI_DATA0_ARQOS),
        .S_AXI_HP0_ARREADY(dpu_eu_0_DPU0_M_AXI_DATA0_ARREADY),
        .S_AXI_HP0_ARSIZE(dpu_eu_0_DPU0_M_AXI_DATA0_ARSIZE),
        .S_AXI_HP0_ARVALID(dpu_eu_0_DPU0_M_AXI_DATA0_ARVALID),
        .S_AXI_HP0_AWADDR(dpu_eu_0_DPU0_M_AXI_DATA0_AWADDR[31:0]),
        .S_AXI_HP0_AWBURST(dpu_eu_0_DPU0_M_AXI_DATA0_AWBURST),
        .S_AXI_HP0_AWCACHE(dpu_eu_0_DPU0_M_AXI_DATA0_AWCACHE),
        .S_AXI_HP0_AWID({1'b0,1'b0,1'b0,1'b0,dpu_eu_0_DPU0_M_AXI_DATA0_AWID}),
        .S_AXI_HP0_AWLEN(dpu_eu_0_DPU0_M_AXI_DATA0_AWLEN),
        .S_AXI_HP0_AWLOCK(dpu_eu_0_DPU0_M_AXI_DATA0_AWLOCK),
        .S_AXI_HP0_AWPROT(dpu_eu_0_DPU0_M_AXI_DATA0_AWPROT),
        .S_AXI_HP0_AWQOS(dpu_eu_0_DPU0_M_AXI_DATA0_AWQOS),
        .S_AXI_HP0_AWREADY(dpu_eu_0_DPU0_M_AXI_DATA0_AWREADY),
        .S_AXI_HP0_AWSIZE(dpu_eu_0_DPU0_M_AXI_DATA0_AWSIZE),
        .S_AXI_HP0_AWVALID(dpu_eu_0_DPU0_M_AXI_DATA0_AWVALID),
        .S_AXI_HP0_BID(dpu_eu_0_DPU0_M_AXI_DATA0_BID),
        .S_AXI_HP0_BREADY(dpu_eu_0_DPU0_M_AXI_DATA0_BREADY),
        .S_AXI_HP0_BRESP(dpu_eu_0_DPU0_M_AXI_DATA0_BRESP),
        .S_AXI_HP0_BVALID(dpu_eu_0_DPU0_M_AXI_DATA0_BVALID),
        .S_AXI_HP0_RDATA(dpu_eu_0_DPU0_M_AXI_DATA0_RDATA),
        .S_AXI_HP0_RDISSUECAP1_EN(1'b0),
        .S_AXI_HP0_RID(dpu_eu_0_DPU0_M_AXI_DATA0_RID),
        .S_AXI_HP0_RLAST(dpu_eu_0_DPU0_M_AXI_DATA0_RLAST),
        .S_AXI_HP0_RREADY(dpu_eu_0_DPU0_M_AXI_DATA0_RREADY),
        .S_AXI_HP0_RRESP(dpu_eu_0_DPU0_M_AXI_DATA0_RRESP),
        .S_AXI_HP0_RVALID(dpu_eu_0_DPU0_M_AXI_DATA0_RVALID),
        .S_AXI_HP0_WDATA(dpu_eu_0_DPU0_M_AXI_DATA0_WDATA),
        .S_AXI_HP0_WID(dpu_eu_0_DPU0_M_AXI_DATA0_WID),
        .S_AXI_HP0_WLAST(dpu_eu_0_DPU0_M_AXI_DATA0_WLAST),
        .S_AXI_HP0_WREADY(dpu_eu_0_DPU0_M_AXI_DATA0_WREADY),
        .S_AXI_HP0_WRISSUECAP1_EN(1'b0),
        .S_AXI_HP0_WSTRB(dpu_eu_0_DPU0_M_AXI_DATA0_WSTRB),
        .S_AXI_HP0_WVALID(dpu_eu_0_DPU0_M_AXI_DATA0_WVALID),
        .S_AXI_HP1_ACLK(clk_wiz_0_clk_out150M),
        .S_AXI_HP1_ARADDR(dpu_eu_0_DPU0_M_AXI_DATA1_ARADDR[31:0]),
        .S_AXI_HP1_ARBURST(dpu_eu_0_DPU0_M_AXI_DATA1_ARBURST),
        .S_AXI_HP1_ARCACHE(dpu_eu_0_DPU0_M_AXI_DATA1_ARCACHE),
        .S_AXI_HP1_ARID({1'b0,1'b0,1'b0,1'b0,dpu_eu_0_DPU0_M_AXI_DATA1_ARID}),
        .S_AXI_HP1_ARLEN(dpu_eu_0_DPU0_M_AXI_DATA1_ARLEN),
        .S_AXI_HP1_ARLOCK(dpu_eu_0_DPU0_M_AXI_DATA1_ARLOCK),
        .S_AXI_HP1_ARPROT(dpu_eu_0_DPU0_M_AXI_DATA1_ARPROT),
        .S_AXI_HP1_ARQOS(dpu_eu_0_DPU0_M_AXI_DATA1_ARQOS),
        .S_AXI_HP1_ARREADY(dpu_eu_0_DPU0_M_AXI_DATA1_ARREADY),
        .S_AXI_HP1_ARSIZE(dpu_eu_0_DPU0_M_AXI_DATA1_ARSIZE),
        .S_AXI_HP1_ARVALID(dpu_eu_0_DPU0_M_AXI_DATA1_ARVALID),
        .S_AXI_HP1_AWADDR(dpu_eu_0_DPU0_M_AXI_DATA1_AWADDR[31:0]),
        .S_AXI_HP1_AWBURST(dpu_eu_0_DPU0_M_AXI_DATA1_AWBURST),
        .S_AXI_HP1_AWCACHE(dpu_eu_0_DPU0_M_AXI_DATA1_AWCACHE),
        .S_AXI_HP1_AWID({1'b0,1'b0,1'b0,1'b0,dpu_eu_0_DPU0_M_AXI_DATA1_AWID}),
        .S_AXI_HP1_AWLEN(dpu_eu_0_DPU0_M_AXI_DATA1_AWLEN),
        .S_AXI_HP1_AWLOCK(dpu_eu_0_DPU0_M_AXI_DATA1_AWLOCK),
        .S_AXI_HP1_AWPROT(dpu_eu_0_DPU0_M_AXI_DATA1_AWPROT),
        .S_AXI_HP1_AWQOS(dpu_eu_0_DPU0_M_AXI_DATA1_AWQOS),
        .S_AXI_HP1_AWREADY(dpu_eu_0_DPU0_M_AXI_DATA1_AWREADY),
        .S_AXI_HP1_AWSIZE(dpu_eu_0_DPU0_M_AXI_DATA1_AWSIZE),
        .S_AXI_HP1_AWVALID(dpu_eu_0_DPU0_M_AXI_DATA1_AWVALID),
        .S_AXI_HP1_BID(dpu_eu_0_DPU0_M_AXI_DATA1_BID),
        .S_AXI_HP1_BREADY(dpu_eu_0_DPU0_M_AXI_DATA1_BREADY),
        .S_AXI_HP1_BRESP(dpu_eu_0_DPU0_M_AXI_DATA1_BRESP),
        .S_AXI_HP1_BVALID(dpu_eu_0_DPU0_M_AXI_DATA1_BVALID),
        .S_AXI_HP1_RDATA(dpu_eu_0_DPU0_M_AXI_DATA1_RDATA),
        .S_AXI_HP1_RDISSUECAP1_EN(1'b0),
        .S_AXI_HP1_RID(dpu_eu_0_DPU0_M_AXI_DATA1_RID),
        .S_AXI_HP1_RLAST(dpu_eu_0_DPU0_M_AXI_DATA1_RLAST),
        .S_AXI_HP1_RREADY(dpu_eu_0_DPU0_M_AXI_DATA1_RREADY),
        .S_AXI_HP1_RRESP(dpu_eu_0_DPU0_M_AXI_DATA1_RRESP),
        .S_AXI_HP1_RVALID(dpu_eu_0_DPU0_M_AXI_DATA1_RVALID),
        .S_AXI_HP1_WDATA(dpu_eu_0_DPU0_M_AXI_DATA1_WDATA),
        .S_AXI_HP1_WID(dpu_eu_0_DPU0_M_AXI_DATA1_WID),
        .S_AXI_HP1_WLAST(dpu_eu_0_DPU0_M_AXI_DATA1_WLAST),
        .S_AXI_HP1_WREADY(dpu_eu_0_DPU0_M_AXI_DATA1_WREADY),
        .S_AXI_HP1_WRISSUECAP1_EN(1'b0),
        .S_AXI_HP1_WSTRB(dpu_eu_0_DPU0_M_AXI_DATA1_WSTRB),
        .S_AXI_HP1_WVALID(dpu_eu_0_DPU0_M_AXI_DATA1_WVALID),
        .S_AXI_HP2_ACLK(clk_wiz_0_clk_out150M),
        .S_AXI_HP2_ARADDR(dpu_eu_0_DPU0_M_AXI_INSTR_ARADDR[31:0]),
        .S_AXI_HP2_ARBURST(dpu_eu_0_DPU0_M_AXI_INSTR_ARBURST),
        .S_AXI_HP2_ARCACHE(dpu_eu_0_DPU0_M_AXI_INSTR_ARCACHE),
        .S_AXI_HP2_ARID({1'b0,1'b0,1'b0,1'b0,dpu_eu_0_DPU0_M_AXI_INSTR_ARID}),
        .S_AXI_HP2_ARLEN(dpu_eu_0_DPU0_M_AXI_INSTR_ARLEN),
        .S_AXI_HP2_ARLOCK(dpu_eu_0_DPU0_M_AXI_INSTR_ARLOCK),
        .S_AXI_HP2_ARPROT(dpu_eu_0_DPU0_M_AXI_INSTR_ARPROT),
        .S_AXI_HP2_ARQOS(dpu_eu_0_DPU0_M_AXI_INSTR_ARQOS),
        .S_AXI_HP2_ARREADY(dpu_eu_0_DPU0_M_AXI_INSTR_ARREADY),
        .S_AXI_HP2_ARSIZE(dpu_eu_0_DPU0_M_AXI_INSTR_ARSIZE),
        .S_AXI_HP2_ARVALID(dpu_eu_0_DPU0_M_AXI_INSTR_ARVALID),
        .S_AXI_HP2_AWADDR(dpu_eu_0_DPU0_M_AXI_INSTR_AWADDR[31:0]),
        .S_AXI_HP2_AWBURST(dpu_eu_0_DPU0_M_AXI_INSTR_AWBURST),
        .S_AXI_HP2_AWCACHE(dpu_eu_0_DPU0_M_AXI_INSTR_AWCACHE),
        .S_AXI_HP2_AWID({1'b0,1'b0,1'b0,1'b0,dpu_eu_0_DPU0_M_AXI_INSTR_AWID}),
        .S_AXI_HP2_AWLEN(dpu_eu_0_DPU0_M_AXI_INSTR_AWLEN),
        .S_AXI_HP2_AWLOCK(dpu_eu_0_DPU0_M_AXI_INSTR_AWLOCK),
        .S_AXI_HP2_AWPROT(dpu_eu_0_DPU0_M_AXI_INSTR_AWPROT),
        .S_AXI_HP2_AWQOS(dpu_eu_0_DPU0_M_AXI_INSTR_AWQOS),
        .S_AXI_HP2_AWREADY(dpu_eu_0_DPU0_M_AXI_INSTR_AWREADY),
        .S_AXI_HP2_AWSIZE(dpu_eu_0_DPU0_M_AXI_INSTR_AWSIZE),
        .S_AXI_HP2_AWVALID(dpu_eu_0_DPU0_M_AXI_INSTR_AWVALID),
        .S_AXI_HP2_BID(dpu_eu_0_DPU0_M_AXI_INSTR_BID),
        .S_AXI_HP2_BREADY(dpu_eu_0_DPU0_M_AXI_INSTR_BREADY),
        .S_AXI_HP2_BRESP(dpu_eu_0_DPU0_M_AXI_INSTR_BRESP),
        .S_AXI_HP2_BVALID(dpu_eu_0_DPU0_M_AXI_INSTR_BVALID),
        .S_AXI_HP2_RDATA(dpu_eu_0_DPU0_M_AXI_INSTR_RDATA),
        .S_AXI_HP2_RDISSUECAP1_EN(1'b0),
        .S_AXI_HP2_RID(dpu_eu_0_DPU0_M_AXI_INSTR_RID),
        .S_AXI_HP2_RLAST(dpu_eu_0_DPU0_M_AXI_INSTR_RLAST),
        .S_AXI_HP2_RREADY(dpu_eu_0_DPU0_M_AXI_INSTR_RREADY),
        .S_AXI_HP2_RRESP(dpu_eu_0_DPU0_M_AXI_INSTR_RRESP),
        .S_AXI_HP2_RVALID(dpu_eu_0_DPU0_M_AXI_INSTR_RVALID),
        .S_AXI_HP2_WDATA(dpu_eu_0_DPU0_M_AXI_INSTR_WDATA),
        .S_AXI_HP2_WID(dpu_eu_0_DPU0_M_AXI_INSTR_WID),
        .S_AXI_HP2_WLAST(dpu_eu_0_DPU0_M_AXI_INSTR_WLAST),
        .S_AXI_HP2_WREADY(dpu_eu_0_DPU0_M_AXI_INSTR_WREADY),
        .S_AXI_HP2_WRISSUECAP1_EN(1'b0),
        .S_AXI_HP2_WSTRB(dpu_eu_0_DPU0_M_AXI_INSTR_WSTRB),
        .S_AXI_HP2_WVALID(dpu_eu_0_DPU0_M_AXI_INSTR_WVALID),
        .USB0_VBUS_PWRFAULT(1'b0));
  zedboard_dpu_xlconcat_0_0 xlconcat_0
       (.In0(dpu_eu_0_dpu_interrupt),
        .In1(1'b0),
        .dout(xlconcat_0_dout));
endmodule
