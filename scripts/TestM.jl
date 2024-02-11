module TM

mutable struct TestM
    teststr::String
    function TestM(str)
        println("TestM init:($str)")
        return new(str)
    end
end

function TestM()
    return TestM("TestM function")
end


end # of module

dt = TM.TestM("Direct")
it = TM.TestM()

println("main: direct: $(dt.teststr) indirect: $(it.teststr)")
