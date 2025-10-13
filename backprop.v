module backprop #(parameter DATAWIDTH=) (
    //ifsd.sd sdif,
    input clk,
    input trigger, //when one, we will calculate the real value
    input [DATAWIDTH-1:0] currW0 [8-1:0][2-1:0],
    input [DATAWIDTH-1:0] currb0 [8-1:0],
    input [DATAWIDTH-1:0] currW1 [3-1:0][8-1:0],
    input [DATAWIDTH-1:0] currb1 [3-1:0],
    input [DATAWIDTH-1:0] inputs [2-1:0],
    input [DATAWIDTH-1:0] reluout [8-1:0],
    input [2-1:0] predictedstate,
    input [2-1:0] realstate,
    input [DATAWIDTH-1:0] softmaxout [3-1:0],
    output reg [DATAWIDTH-1:0] newW0 [8-1:0][2-1:0],
    output reg [DATAWIDTH-1:0] newb0 [9-1:0],
    output reg [DATAWIDTH-1:0] newW1 [3-1:0][8-1:0],
    output reg [DATAWIDTH-1:0] newb1 [3-1:0],
);








reg [DATAWIDTH-1:0] dldZ1 [3-1:0][1-1:0];

/*
    creates dl/dZ1 (3, 1):
    - take softmax output and subtract 1 for the real state
*/
genvar i;
generate 
    for(i = 0; i < 3; i = i + 1) begin gen_loop
        always @(posedge clk) begin 
            if (realstate == i) dldZ1[i][0] = softmaxout[i] - 1;
            else dldZ1[i][0] = softmaxout[i]
        end
    end
endgenerate

/*
    creates dl/dW1 (3, 9)
    - dl/dW1 = dl/dZ1 * dZ1/dW1
        - dl/dZ1 (3, 1): calculated above
        - dZ1/dW1 (1, 9): transpose of hidden layer output plus bias
*/
//create dZ1/dW1
reg [DATAWIDTH-1:0] dZ1dW1 [1-1:0][9-1:0];
genvar j;
generate 
    for(j = 0; j < 9-1; j = j + 1) begin gen_loop
        always @(posedge clk) begin 
            dZ1dW1[0][j] = hiddenlayerout[j];
        end
    end
endgenerate
//last dZ1dW1 entry for the bias
always @(posedge clk) begin 
    dZ1dW1[0][8] = 1;
end

//create dl/dW1
wire [DATAWIDTH-1:0] dldW1 [3-1:0][9-1:0];
matmul #(.DATAWIDTH(DATAWIDTH), .M(3), N(1), .P(9)) matmuldldW1 (
    .clk(clk),
    .A(dldZ1),
    .B(dZ1dW1),
    .C(dldW1)
)


/*
    given dl/dW1, update W1 and b1
*/
genvar k, l;
generate 
    for(k = 0; k < 3; k = k + 1) begin looprow_newW1
        for (l = 0; l < 8; l = l + 1) begin loopcol_newW1
            newW1[k][l] = currW1[k][l] - dldW1[k][l] >> 10
        end
        newb1[k] = currb1[k] - dldW1[k][9-1] >> 10
    end
endgenerate

/*
    create dl/dA0 which is gradiant loss with respect to output of hidden layer after relu
*/
//create currW1reshape to be currW1 but transposed to 8 by 3
reg [DATAWIDTH-1:0] currW1reshape [8-1:0][3-1:0];
wire [DATAWIDTH-1:0] dldA0 [8-1:0][1-1:0];

genvar m, n;
generate 
    for(m = 0; m < 8; m = m + 1) begin looprow_reshapeW1 
        for (n = 0; n < 3; n = n + 1) begin loopcol_reshapeW1 
            currW1reshape[m][n] = currW1[n][m]
        end
    end
endgenerate
matmul #(.DATAWIDTH(DATAWIDTH), .M(8), N(3), .P(1)) matmuldldA0 (
    .clk(clk),
    .A(currW1reshape),
    .B(dldZ1),
    .C(dldA0)
)

/*
    create dl/dZ0 which is gradiant loss with respect to output of hidden layer before relu
*/
reg [DATAWIDTH-1:0] dldZ0 [8-1:0][1-1:0];

genvar o;
generate 
    for(o = 0; o < 8; o = o + 1) begin loop_dldZ0
        always @(posedge clk) begin 
            if (reluout[o][0] > 0)
                dldZ0[o][0] = dldA0[o][0];
            else
                dldZ0[o][0] = 0;
        end
    end
endgenerate

/*
    creates dl/dW0 (8, 3)
    - dl/dW0 = dl/dZ0 * dZ0/dW0
        - dl/dZ0 (8, 1): calculated above
        - dZ0/dW0 (1, 3): inputs plus bias (1)
*/
//create dZ0dW0
wire [DATAWIDTH-1:0] dZ0dW0 [1-1:0][3-1:0];
genvar p;
generate 
    for(p = 0; p < 2; p = p + 1) begin 
        dZdW0[0][p] = inputs[p];
    end
    dZdW0[0][2] = 1;
endgenerate

//create dldW0
wire [DATAWIDTH-1:0] dldW0 [8-1:0][3-1:0];

matmul #(.DATAWIDTH(DATAWIDTH), .M(8), N(1), .P(3)) matmuldldW0 (
    .clk(clk),
    .A(dldZ0),
    .B(dZ0dW0),
    .C(dlDW0)
)


/*
    given dl/dW0, update W0 and b0
*/
genvar q, r;
generate 
    for(q = 0; q < 8; q = q + 1) begin looprow_newW0
        for (r = 0; r < 2; r = r + 1) begin loopcol_newW0
            newW0[q][r] = currW0[q][r] - dldW0[q][r] >> 10
        end
        newb1[q] = currb1[q] - dldW1[q][2] >> 10
    end
endgenerate

endmodule