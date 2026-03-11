while(<>){
 # Capture start time from "Server port:" line
 my ($start,$host,) = ($1,$2,) if /\w{3} \d{2} \w{3} \d{4} (\d{2}:\d{2}:\d{2}) \w+ UTC (\w+\S*) Starting vLLM server/;
 if ($start && $host) {
   # print "START: $start $host\n";
   my $key = $host;
   $h{$key}->[0] = $start;
 }
 
 # Capture ready time, host, from "vLLM server is ready" line
 my($ready, $host2, ) = ($1, $2,) if /\w{3} \d{2} \w{3} \d{4} (\d{2}:\d{2}:\d{2}) \w+ UTC (\w+\S*) vLLM server is ready/;
 if ($ready && $host2) {
   # print "READY: $ready $host2\n";
   my $key = $host2;
   $h{$key}->[1] = $ready;
 }
}

foreach my $k (sort keys(%h)) {
  print "$k\t$h{$k}->[0]\t$h{$k}->[1]\n";
}
