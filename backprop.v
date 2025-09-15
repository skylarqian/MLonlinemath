module backprop #(parameter DATAWIDTH=) (
    input clk,
    input trigger, //when one, we will calculate the real value
    input [DATAWIDTH-1:0] currW0 [8-1:0][2-1:0],
    input [DATAWIDTH-1:0] currb0 [8-1:0],
    input [DATAWIDTH-1:0] currW1 [3-1:0][8-1:0],
    input [DATAWIDTH-1:0] currb1 [3-1:0],
    input [2-1:0] predictedstate,
    input [2-1:0] realstate,
    input [DATAWIDTH-1:0] softmaxout [3-1:0],
    input [DATAWIDTH-1:0] hiddenlayerout [8-1:0],
    output reg [DATAWIDTH-1:0] newW0 [8-1:0][2-1:0],
    output reg [DATAWIDTH-1:0] newb0 [9-1:0],
    output reg [DATAWIDTH-1:0] newW1 [3-1:0][8-1:0],
    output [reg DATAWIDTH-1:0] newb1 [3-1:0],
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
        - dZ1/dW1 (1, 9): transpose of hidden layer output plus 
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
            newW1[k][l] = currW1[k][l] - dlDw1[k][l] >> 10
        end
        newb1[k] = currb1[k] - dlDw1[k][9-1]
    end
endgenerate

/*
    create dl/dA0 which is gradiant loss with respect to output of hidden layer
*/
endmodule