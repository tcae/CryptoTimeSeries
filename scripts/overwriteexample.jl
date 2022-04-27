
module PrintTest

struct MyType
    el::Int32
end

Base.show(io::IO, mystruct::MyType) = println("My print: $(mystruct.el)")


end

ms = PrintTest.MyType(42)
print(ms)
