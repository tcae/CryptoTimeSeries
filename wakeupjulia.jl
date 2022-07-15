using ProgressLogging

@progress for i in 1:10
    sleep(1)
end

println("wake up julia $VERSION")

