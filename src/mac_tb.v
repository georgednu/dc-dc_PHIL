`timescale 1ns/1ps

// -----------------------------------------------------------------------------
// mac_36x25_tb.v
//
// Tests:
//   1. Single multiply:        1.0   * 1.0   = 1.0
//   2. Negative multiply:     -1.0   * 1.0   = -1.0
//   3. Fractional multiply:    0.5   * 0.5   = 0.25
//   4. 4-term accumulation
//   5. 256-term accumulation   (max ACC_GUARD=8 depth)
//   6. Overflow detection
//   7. Max negative value
//   8. Two-stream mux          (sel_in toggles between streams)
// -----------------------------------------------------------------------------

module mac_36x25_tb;

    // -------------------------------------------------------------------------
    // DUT signals
    // -------------------------------------------------------------------------
    reg         clk, rst;
    reg         first_in, valid_in, sel_in;
    reg  signed [35:0] a_in;
    reg  signed [24:0] b_in;

    wire signed [35:0] p_out;
    wire               overflow;
    wire               valid_out;
    wire               sel_out;

    mac_36x25 #(
        .ACC_GUARD (8)
    ) dut (
        .clk       (clk),
        .rst       (rst),
        .a_in      (a_in),
        .b_in      (b_in),
        .first_in  (first_in),
        .valid_in  (valid_in),
        .sel_in    (sel_in),
        .p_out     (p_out),
        .overflow  (overflow),
        .valid_out (valid_out),
        .sel_out   (sel_out)
    );

    // -------------------------------------------------------------------------
    // Clock — 100 MHz
    // -------------------------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    // -------------------------------------------------------------------------
    // Fixed-point conversion functions
    // Q12.23: 36-bit signed, scale = 2^23
    // Q1.23:  25-bit signed, scale = 2^23
    // -------------------------------------------------------------------------
    function automatic signed [35:0] to_q1223;
        input real val;
        to_q1223 = $rtoi(val * (2.0 ** 23));
    endfunction

    function automatic signed [24:0] to_q123;
        input real val;
        to_q123 = $rtoi(val * (2.0 ** 23));
    endfunction

    function automatic real from_q1223;
        input signed [35:0] bits;
        from_q1223 = $itor(bits) / (2.0 ** 23);
    endfunction

    // -------------------------------------------------------------------------
    // Feed one cycle of input to the MAC
    // -------------------------------------------------------------------------
    task automatic feed;
        input real  a_val;
        input real  b_val;
        input       is_first;
        input       sel;
        begin
            a_in     = to_q1223(a_val);
            b_in     = to_q123(b_val);
            first_in = is_first;
            valid_in = 1'b1;
            sel_in   = sel;
            @(posedge clk); #1;
            first_in = 1'b0;
        end
    endtask

    // De-assert valid between accumulations
    task automatic idle;
        input integer n;
        integer k;
        begin
            valid_in = 1'b0;
            first_in = 1'b0;
            a_in     = '0;
            b_in     = '0;
            for (k = 0; k < n; k = k + 1)
                @(posedge clk); #1;
        end
    endtask

    // Wait for pipeline to flush (5 cycles) then sample and display result
    // Pass expected real value and test label
    task automatic check;
        input real   expected;
        input string label;
        input        expect_overflow;
        real         got;
        real         err;
        begin
            repeat(5) @(posedge clk); #1;
            got = from_q1223(p_out);
            err = got - expected;
            if (err < 0.0) err = -err;

            $display("[%-30s] expected=%12.6f  got=%12.6f  err=%e  overflow=%b(%s)",
                label,
                expected,
                got,
                err,
                overflow,
                expect_overflow ? "expected" : "unexpected"
            );

            // Warn on large error (> 1 LSB = 2^-23)
            if (err > (1.0 / (2.0**23)) && !expect_overflow)
                $display("  *** WARNING: error exceeds 1 LSB ***");

            if (overflow != expect_overflow)
                $display("  *** WARNING: overflow flag mismatch ***");
        end
    endtask

    // -------------------------------------------------------------------------
    // Stimulus
    // -------------------------------------------------------------------------
    real acc;
    integer i;

    initial begin
        $dumpfile("mac_36x25_tb.vcd");
        $dumpvars(0, mac_36x25_tb);

        // Reset
        rst      = 1'b1;
        first_in = 1'b0;
        valid_in = 1'b0;
        sel_in   = 1'b0;
        a_in     = '0;
        b_in     = '0;
        repeat(4) @(posedge clk);
        rst = 1'b0;
        @(posedge clk); #1;

        $display("========================================");
        $display(" MAC 36x25 Testbench (Q12.23 x Q1.23) ");
        $display("========================================");

        // --------------------------------------------------------------------
        // Test 1: 1.0 * 1.0 = 1.0
        // --------------------------------------------------------------------
        feed(1.0, 1.0, 1, 0);
        idle(0);
        check(1.0, "1.0 * 1.0", 0);

        // --------------------------------------------------------------------
        // Test 2: -1.0 * 1.0 = -1.0
        // --------------------------------------------------------------------
        feed(-1.0, 1.0, 1, 0);
        idle(0);
        check(-1.0, "-1.0 * 1.0", 0);

        // --------------------------------------------------------------------
        // Test 3: 0.5 * 0.5 = 0.25
        // --------------------------------------------------------------------
        feed(0.5, 0.5, 1, 0);
        idle(0);
        check(0.25, "0.5 * 0.5", 0);

        // --------------------------------------------------------------------
        // Test 4: 4-term accumulation
        //   2.0*0.5 + (-1.5)*0.25 + 3.0*0.125 + (-0.5)*0.5
        //   = 1.0 - 0.375 + 0.375 - 0.25 = 0.75
        // --------------------------------------------------------------------
        acc = 0.0;
        feed( 2.0,   0.5,   1, 0); acc  =  2.0   *  0.5;
        feed(-1.5,   0.25,  0, 0); acc += -1.5   *  0.25;
        feed( 3.0,   0.125, 0, 0); acc +=  3.0   *  0.125;
        feed(-0.5,   0.5,   0, 0); acc += -0.5   *  0.5;
        idle(0);
        check(acc, "4-term accumulation", 0);

        // --------------------------------------------------------------------
        // Test 5: 256-term accumulation (max depth for ACC_GUARD=8)
        //   256 * (0.5 * 0.5) = 256 * 0.25 = 64.0
        // --------------------------------------------------------------------
        acc = 0.0;
        for (i = 0; i < 256; i = i + 1) begin
            feed(0.5, 0.5, i == 0, 0);
            acc += 0.5 * 0.5;
        end
        idle(0);
        check(acc, "256-term accumulation", 0);

        // --------------------------------------------------------------------
        // Test 6: Large value, no overflow
        //   4000.0 * 0.999 ~ 3996.0 — within Q12.23 max (~4096)
        // --------------------------------------------------------------------
        feed(4000.0, 0.999, 1, 0);
        idle(0);
        check(4000.0 * 0.999, "large value no overflow", 0);

        // --------------------------------------------------------------------
        // Test 7: Overflow detection
        //   4000.0 * 0.999 + 4000.0 * 0.999 ~ 7992 > 4096 (Q12.23 max)
        // --------------------------------------------------------------------
        feed(4000.0, 0.999, 1, 0);
        feed(4000.0, 0.999, 0, 0);
        idle(0);
        repeat(5) @(posedge clk); #1;
        $display("[%-30s] overflow=%b (expected 1)",
            "overflow detection", overflow);
        if (!overflow)
            $display("  *** WARNING: overflow not detected ***");

        // --------------------------------------------------------------------
        // Test 8: Max negative — -4096.0 * 0.999 ~ -4092
        // --------------------------------------------------------------------
        feed(-4096.0, 0.999, 1, 0);
        idle(0);
        check(-4096.0 * 0.999, "max negative", 0);

        // --------------------------------------------------------------------
        // Test 9: Two-stream mux
        //   Stream 0: 1.0 * 0.5 = 0.5  (sel=0)
        //   Stream 1: 2.0 * 0.5 = 1.0  (sel=1)
        //   Feed interleaved — sel tags which stream result belongs to
        //   Pipeline delay means results come out 5 cycles after input
        // --------------------------------------------------------------------
        $display("-- Two-stream mux test --");

        // Feed stream 0 then stream 1 back to back
        feed(1.0, 0.5, 1, 0);   // stream 0, first
        feed(2.0, 0.5, 1, 1);   // stream 1, first (new accumulation)
        idle(0);

        // Wait and sample both results
        // Stream 0 result arrives 5 cycles after its input (cycle 0)
        // Stream 1 result arrives 5 cycles after its input (cycle 1)
        repeat(5) @(posedge clk); #1;
        $display("[stream mux sel_out=%b] got=%12.6f (expected stream 0 = 0.500000)",
            sel_out, from_q1223(p_out));
        @(posedge clk); #1;
        $display("[stream mux sel_out=%b] got=%12.6f (expected stream 1 = 1.000000)",
            sel_out, from_q1223(p_out));

        $display("========================================");
        $display(" Done.");
        $display("========================================");
        $finish;
    end

endmodule