using Blink, TableView

function update_cell(arr, msg)
    row = msg["row"] + 1
    col = parse(Int, match(r"Column(\d+)", msg["col"])[1])
    arr[row, col] = parse(eltype(arr), msg["new"])
end

x = rand(10,10)

# mock-up loop of Julia program not terminating to the REPL until user decides so
leave = false
while !leave
    w = Blink.Window();
    body!(w, TableView.showtable(x, cell_changed = msg -> update_cell(x, msg)))
    println("Enter to continue or Y to leave: ")
    str = readline()
    if str == "y" || str == "Y"
       leave = true
    end
end