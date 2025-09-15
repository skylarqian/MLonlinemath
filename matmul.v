module matmul #(parameter DATAWIDTH= M= N= P=) (
    input clk,
    input signed [DATAWIDTH-1:0] A [M-1:0] [N-1:0],
    input signed [DATAWIDTH-1:0] B [N-1:0] [P-1:0],
    output reg signed [DATAWIDTH-1:0] C [M-1:0] [P-1:0]
);

integer row, col;
reg [DATAWIDTH-1:0] sum;
always @(posedge clk) begin
    for(row = 0; row < M; row = row + 1) begin
        for(col = 0; col < P; col = col + 1) begin
            sum = 0;
            for(idx = 0; idx < N; idx + 1) begin 
                //
                sum += A[row][idx]*B[idx][col]
            end
            C[row][col] = sum;
        end
    end
end
endmodule