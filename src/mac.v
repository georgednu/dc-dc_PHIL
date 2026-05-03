`timescale 1ns/1ps

module mac_36x25 #(
    parameter ACC_GUARD = 8                         // log2(max accumulations)
)(
    input  wire                clk,
    input  wire                rst, en,

    // Data
    input  wire signed [35:0]  a_in,               // Q12.23
    input  wire signed [24:0]  b_in,               // Q1.23

    // Control
    input  wire                first_in,
    input  wire                valid_in,
    // Output
    output reg  signed [35:0]  p_out,
    output reg                 valid_out,
);


wire signed [17:0] a_L;
wire signed [17:0] a_H;

assign a_L = (valid_in)? {a_in[35], a_in[16:0]} : 0;
assign a_H = (valid_in)? {a_in[35], a_in[33:17]} : 0;

reg signed [17:0] mult_ina_L;
reg signed [17:0] mult_ina_H;

reg signed [24:0] mult_inb_L;
reg signed [24:0] mult_inb_H;

reg signed [42:0] mult_out_L;
reg signed [42:0] mult_out_H;

reg signed [42:0] mult_out;
always @(posedge clk) begin
    if (!rst) begin
        mult_ina_L<=0;
        mult_ina_H<=0;
        mult_inb_L<=0;
        mult_inb_H<=0;
        mult_out_L<=0;
        mult_out_H<=0;
        mult_out<=0;
    end
    else if (en && valid_in) begin
        mult_ina_L<=a_L;
        mult_ina_H<=a_H;
        mult_inb_L<=b_in;
        mult_inb_H<=b_in;
        
        mult_out_L <= mult_ina_L*mult_inb_L;
        mult_out_H <= mult_ina_H*mult_inb*H;

        mult_out <= mult_out_H + mult_out_L[42:17];
    end
end

reg [3:0] first_in_pipeline;
always @(posedge clk) begin
    if (!rst)
        first_in_pipeline <= 0;
    else if (en && valid_in)
        first_in_pipeline <= {first_in_pipeline[2:0], first_in};
end


reg signed [42:0] mac_in;

always @(posedge clk) begin
    if (!rst) begin
        mac_in<=0;
        p_out<=0;
    end else if (en && valid_in) begin
        mac_in <= mult_out;
        if (first_in_pipeline[3])
            p_out <= mac_in;
        else
            p_out <= p_out + mac_in;
    end
end




endmodule