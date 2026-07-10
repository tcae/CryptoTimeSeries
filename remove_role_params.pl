#!/usr/bin/perl
use strict;
use warnings;

my @files = glob("scripts/*.jl Trade/src/*.jl Xch/src/*.jl");

foreach my $file (@files) {
    next unless -f $file;
    
    open my $fh, '<', $file or die "Cannot open $file: $!";
    my @lines = <$fh>;
    close $fh;
    
    my @output;
    foreach my $line (@lines) {
        # Remove '; role=Xch.trade_exchange_spot)' and similar patterns
        $line =~ s/;\s*role\s*=\s*\w+[\w.]*\s*\)/\)/g;
        $line =~ s/;\s*role\s*=\s*\w+[\w.]*\s*,\)/,\)/g;
        push @output, $line;
    }
    
    open $fh, '>', $file or die "Cannot write $file: $!";
    print $fh @output;
    close $fh;
    
    print "Updated $file\n";
}
