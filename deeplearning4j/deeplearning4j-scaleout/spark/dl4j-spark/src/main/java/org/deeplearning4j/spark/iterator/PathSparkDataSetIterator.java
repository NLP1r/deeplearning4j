package org.deeplearning4j.spark.iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.deeplearning4j.api.loader.DataSetLoader;
import org.deeplearning4j.spark.data.loader.RemoteFileSource;
import org.nd4j.linalg.dataset.DataSet;

import java.io.OutputStream;
import java.net.URI;
import java.util.Collection;
import java.util.Iterator;

/**
 * A DataSetIterator that loads serialized DataSet objects (saved with {@link DataSet#save(OutputStream)}) from
 * a String that represents the path (for example, on HDFS)
 *
 * @author Alex Black
 */
public class PathSparkDataSetIterator extends BaseDataSetIterator<String> {

    public static final int BUFFER_SIZE = 4194304; //4 MB
    private FileSystem fileSystem;
    private DataSetLoader dataSetLoader;

    public PathSparkDataSetIterator(Iterator<String> iter, DataSetLoader dataSetLoader) {
        this.dataSetStreams = null;
        this.iter = iter;
        this.dataSetLoader = dataSetLoader;
    }

    public PathSparkDataSetIterator(Collection<String> dataSetStreams, DataSetLoader dataSetLoader) {
        this.dataSetStreams = dataSetStreams;
        iter = dataSetStreams.iterator();
        this.dataSetLoader = dataSetLoader;
    }

    @Override
    public DataSet next() {
        DataSet ds;
        if (preloadedDataSet != null) {
            ds = preloadedDataSet;
            preloadedDataSet = null;
        } else {
            ds = load(iter.next());
        }

        // FIXME: int cast
        totalOutcomes = ds.getLabels() == null ? 0 : (int) ds.getLabels().size(1); //May be null for layerwise pretraining
        inputColumns = (int) ds.getFeatureMatrix().size(1);
        batch = ds.numExamples();

        if (preprocessor != null)
            preprocessor.preProcess(ds);
        return ds;
    }

    protected synchronized DataSet load(String path) {
        if (fileSystem == null) {
            try {
                fileSystem = FileSystem.get(new URI(path), new Configuration());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
        cursor++;
        try{
            return dataSetLoader.load(new RemoteFileSource(path, fileSystem, BUFFER_SIZE));
        } catch (Exception e){
            throw new RuntimeException("Error loading DataSet at path " + path + " - DataSet may be corrupt or invalid." +
                    " Spark DataSets can be validated using org.deeplearning4j.spark.util.data.SparkDataValidation", e);
        }
    }
}
