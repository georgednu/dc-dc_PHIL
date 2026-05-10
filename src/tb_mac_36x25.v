`timescale 1ns/1ps
`include "mac.v"

module tb_mac_36x25;

// ============================================================
// Fixed-point format
// a_in : Q12.23 -- 36-bit signed, 23 fractional bits
// b_in : Q1.23  -- 25-bit signed, 23 fractional bits
// p_out: Q12.23 -- 36-bit signed, 23 fractional bits
//
// Pipeline latency: 5 cycles from valid_in to p_out
//   Cycle 0: mult_ina/b registered
//   Cycle 1: mult_out_L/H registered (multiply)
//   Cycle 2: mult_out registered (combine)
//   Cycle 3: mac_in registered
//   Cycle 4: p_out written, valid_out high
// ============================================================

localparam CLK_HALF = 5; // 10 ns period, 100 MHz

// Q12.23 constants (raw integer = real * 2^23)
// 1.0  = 8388608  = 23'h800000
// 0.5  = 4194304  = 22'h400000
// 2.0  = 16777216 = 24'h1000000
// 3.0  = 25165824 = 24'h1800000
localparam signed [35:0] ONE_A   = 36'sh800000;
localparam signed [24:0] ONE_B   = 25'sh800000;
localparam signed [35:0] HALF_A  = 36'sh400000;
localparam signed [35:0] TWO_A   = 36'sh1000000;
localparam signed [35:0] THREE_A = 36'sh1800000;

// ============================================================
// DUT signals
// ============================================================
reg              clk, rst, en;
reg  signed [35:0] a_in;
reg  signed [24:0] b_in;
reg              first_in, valid_in;
wire signed [35:0] p_out;
wire             valid_out;

// ============================================================
// Scoreboard
// ============================================================
integer pass_count;
integer fail_count;

// ============================================================
// DUT
// ============================================================
mac_36x25 dut (
    .clk      (clk),
    .rst      (rst),
    .en       (en),
    .a_in     (a_in),
    .b_in     (b_in),
    .first_in (first_in),
    .valid_in (valid_in),
    .p_out    (p_out),
    .valid_out(valid_out)
);

// ============================================================
// Clock
// ============================================================
initial clk = 0;
always #CLK_HALF clk = ~clk;

// ============================================================
// Waveform dump
// ============================================================
initial begin
    $dumpfile("tb_mac_36x25.vcd");
    $dumpvars(0, tb_mac_36x25);
end

// ============================================================
// Tasks
// No [127:0] string args (SV only).
// No localparam expressions inside task arguments (V2001).
// Unicode removed from comments.
// ============================================================

task drive_sample;
    input signed [35:0] a;
    input signed [24:0] b;
    input               is_first;
    begin
        @(negedge clk);
        a_in     = a;
        b_in     = b;
        first_in = is_first;
        valid_in = 1;
        @(posedge clk);
    end
endtask

task idle_one;
    begin
        @(negedge clk);
        valid_in = 0;
        first_in = 0;
        @(posedge clk);
    end
endtask

// drain: 6 idle cycles (latency 5 + 1 margin)
task drain;
    begin
        idle_one; idle_one; idle_one;
        idle_one; idle_one; idle_one;
    end
endtask

task do_reset;
    begin
        @(negedge clk);
        rst      = 0;
        en       = 0;
        a_in     = 0;
        b_in     = 0;
        first_in = 0;
        valid_in = 0;
        @(posedge clk);
        @(posedge clk);
        @(negedge clk);
        rst = 1;
        en  = 1;
    end
endtask

task check_result;
    input signed [35:0] expected;
    input integer       tol;
    integer             diff;
    begin
        @(negedge clk);
        diff = $signed(p_out) - $signed(expected);
        if (diff < 0) diff = -diff;
        if (diff <= tol) begin
            $display("  [PASS] expected=%0d  got=%0d", $signed(expected), $signed(p_out));
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] expected=%0d  got=%0d", $signed(expected), $signed(p_out));
            fail_count = fail_count + 1;
        end
    end
endtask

task check_valid_low;
    begin
        @(posedge clk); #1;
        if (valid_out === 1'b0) begin
            $display("  [PASS] valid_out=0 as expected");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] valid_out should be 0, got %0b", valid_out);
            fail_count = fail_count + 1;
        end
    end
endtask

task check_valid_high;
    begin
        @(posedge clk); #1;
        if (valid_out === 1'b1) begin
            $display("  [PASS] valid_out=1 as expected");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] valid_out should be 1, got %0b", valid_out);
            fail_count = fail_count + 1;
        end
    end
endtask

// ============================================================
// Main stimulus
// ============================================================
initial begin

    pass_count = 0;
    fail_count = 0;

    $display("========================================");
    $display("  tb_mac_36x25");
    $display("========================================");

    do_reset;

    // ----------------------------------------------------------
    // T1: Single multiply -- 1.0 x 1.0 = 1.0
    //     Q12.23: 1.0 = 36'sh800000
    // ----------------------------------------------------------
    $display("\n--- T1: 1.0 x 1.0 = 1.0 ---");
    drive_sample(ONE_A, ONE_B, 1);
    drain;
    check_result(36'sh800000, 8);

    do_reset;

    // ----------------------------------------------------------
    // T2: Accumulate 4 terms -- 0.5 x 1.0 each = 2.0 total
    // ----------------------------------------------------------
    $display("\n--- T2: 4x(0.5 x 1.0) = 2.0 ---");
    drive_sample(HALF_A, ONE_B, 1);
    drive_sample(HALF_A, ONE_B, 0);
    drive_sample(HALF_A, ONE_B, 0);
    drive_sample(HALF_A, ONE_B, 0);
    drain;
    check_result(36'sh1000000, 8);

    do_reset;

    // ----------------------------------------------------------
    // T3: Negative operand -- (-1.0) x 1.0 = -1.0
    // ----------------------------------------------------------
    $display("\n--- T3: (-1.0) x 1.0 = -1.0 ---");
    drive_sample(-ONE_A, ONE_B, 1);
    drain;
    check_result(-36'sh800000, 8);

    do_reset;

    // ----------------------------------------------------------
    // T4: valid_out timing
    //     1 cycle of valid_in, then idle.
    //     valid_out must be 0 during latency, 1 at cycle 5, 0 after.
    // ----------------------------------------------------------
    $display("\n--- T4a: valid_out low before latency ---");
    drive_sample(ONE_A, ONE_B, 1);
    idle_one; idle_one; idle_one; idle_one;
    check_valid_low;

    $display("--- T4b: valid_out high at latency boundary ---");
    @(posedge clk);
    check_valid_high;

    $display("--- T4c: valid_out low after single-cycle burst ---");
    @(posedge clk);
    check_valid_low;

    do_reset;

    // ----------------------------------------------------------
    // T5: en=0 freezes pipeline -- p_out must stay 0
    // ----------------------------------------------------------
    $display("\n--- T5: en=0 freezes pipeline ---");
    drive_sample(TWO_A, ONE_B, 1);
    @(negedge clk); en = 0; valid_in = 0;
    idle_one; idle_one; idle_one;
    idle_one; idle_one; idle_one; idle_one;
    @(negedge clk);
    if (p_out === 36'sh0) begin
        $display("  [PASS] p_out held at 0 while en=0");
        pass_count = pass_count + 1;
    end else begin
        $display("  [FAIL] p_out changed while en=0, got %0d", $signed(p_out));
        fail_count = fail_count + 1;
    end
    @(negedge clk); en = 1;

    do_reset;

    // ----------------------------------------------------------
    // T6: first_in restart mid-accumulation
    //     2 cycles 1.0 x 1.0, then first_in=1 with 3.0 x 1.0
    //     expected = 3.0, not 5.0
    // ----------------------------------------------------------
    $display("\n--- T6: first_in restart mid-accumulate ---");
    drive_sample(ONE_A,   ONE_B, 1);
    drive_sample(ONE_A,   ONE_B, 0);
    drive_sample(THREE_A, ONE_B, 1);
    drain;
    check_result(36'sh1800000, 8);

    do_reset;

    // ----------------------------------------------------------
    // T7: bubble mid-burst
    //     4 real samples with 1 bubble in the middle.
    //     All 4 samples must accumulate; bubble must not corrupt.
    // ----------------------------------------------------------
    $display("\n--- T7: bubble mid-burst ---");
    drive_sample(HALF_A, ONE_B, 1);
    drive_sample(HALF_A, ONE_B, 0);
    idle_one;
    drive_sample(HALF_A, ONE_B, 0);
    drive_sample(HALF_A, ONE_B, 0);
    drain;
    check_result(36'sh1000000, 8);

    do_reset;

    // ----------------------------------------------------------
    // T8: reset mid-pipeline clears p_out and valid_out
    // ----------------------------------------------------------
    $display("\n--- T8: reset mid-pipeline ---");
    drive_sample(ONE_A, ONE_B, 1);
    drive_sample(ONE_A, ONE_B, 0);
    @(negedge clk); rst = 0;
    @(posedge clk);
    @(posedge clk);
    @(negedge clk);
    if (p_out === 36'sh0 && valid_out === 1'b0) begin
        $display("  [PASS] reset cleared p_out and valid_out");
        pass_count = pass_count + 1;
    end else begin
        $display("  [FAIL] after reset: p_out=%0d valid_out=%0b",
                 $signed(p_out), valid_out);
        fail_count = fail_count + 1;
    end

    // ----------------------------------------------------------
    // Summary
    // ----------------------------------------------------------
    $display("\n========================================");
    $display("  %0d passed,  %0d failed", pass_count, fail_count);
    $display("========================================");
    if (fail_count == 0)
        $display("  ALL TESTS PASSED");
    else
        $display("  SOME TESTS FAILED");

    $finish;
end

// ============================================================
// Timeout watchdog
// ============================================================
initial begin
    #500000;
    $display("[TIMEOUT] simulation exceeded 500 us");
    $finish;
end

endmodule